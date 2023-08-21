import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from typing import Union
from PIL import Image

from diffusers import (
    FlaxAutoencoderKL,
    FlaxDPMSolverMultistepScheduler,
    FlaxStableDiffusionPipeline,
    FlaxUNet2DConditionModel,
)

from transformers import CLIPFeatureExtractor, CLIPTokenizer, FlaxCLIPTextModel, FlaxCLIPTextModelWithProjection

import diffusers.schedulers.scheduling_ddim_flax

from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

# typehint definition
transformed_params = dict
params = dict
rng = jax.random.PRNGKey
noise_scheduler_state = diffusers.schedulers.scheduling_ddim_flax.DDIMSchedulerState

sharding = PositionalSharding(mesh_utils.create_device_mesh((jax.device_count(),)))


def save_model_shape_as_json(file_name: str, params: dict) -> None:
    shape = jax.tree_map(lambda x: x.shape, params)
    import json

    # Save the dictionary as JSON
    with open(file_name, "w") as json_file:
        json.dump(shape, json_file)


def shard_weight_column(model_params: jnp.array):
    device_count = jax.device_count()

    # get parallel device count to slice weight column wise
    sharding = PositionalSharding(mesh_utils.create_device_mesh((device_count,)))

    # check if model params is divisible by shard if it's not then just replicate for now
    if model_params.shape[-1] % device_count == 0:
        # replicate on last axis because sd last axis is shardable

        param_dim_count = len(model_params.shape)

        if param_dim_count > 1:

            # just putting 1 as placeholder
            # example [1,1,1,8] which replicate

            sharding_rule = sharding.reshape([1] * (param_dim_count - 1) + [device_count])

            model_params = jax.device_put(model_params, sharding_rule)

        else:
            model_params = jax.device_put(model_params, sharding.replicate())
        pass
    else:
        # just replicate everything on all devices
        model_params = jax.device_put(model_params, sharding.replicate())

    return model_params


model_dir = "/home/teor/secondary_storage/SDXL-sharding/stable-diffusion-xl-base-1.0-flax"
# load the model params and model object

tokenizer = CLIPTokenizer.from_pretrained(model_dir, subfolder="tokenizer")
tokenizer_2 = CLIPTokenizer.from_pretrained(model_dir, subfolder="tokenizer_2")

text_encoder, text_encoder_params = FlaxCLIPTextModel.from_pretrained(
    model_dir, subfolder="text_encoder", dtype=jnp.bfloat16, _do_init=False
)

text_encoder_2, text_encoder_params_2 = FlaxCLIPTextModelWithProjection.from_pretrained(
    model_dir, subfolder="text_encoder_2", dtype=jnp.bfloat16, _do_init=False
)

unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
    model_dir, subfolder="unet", dtype=jnp.bfloat16, use_memory_efficient=False
)

vae, vae_params = FlaxAutoencoderKL.from_pretrained(
    model_dir,
    dtype=jnp.bfloat16,
    subfolder="vae",
)

noise_scheduler, scheduler_params = FlaxDPMSolverMultistepScheduler.from_pretrained(model_dir,subfolder="scheduler")
# noise_scheduler = FlaxDPMSolverMultistepScheduler(
#     beta_start=0.00085,
#     beta_end=0.012,
#     beta_schedule="scaled_linear",
#     num_train_timesteps=1000,
# )

# scheduler_params = noise_scheduler.create_state()

# shard the weights across device
unet_params = jax.tree_map(shard_weight_column, unet_params)
text_encoder_params = jax.tree_map(shard_weight_column, text_encoder_params)
text_encoder_params_2 = jax.tree_map(shard_weight_column, text_encoder_params_2)
vae_params = jax.tree_map(shard_weight_column, vae_params)
# unet_params = jax.tree_map(lambda x: jax.device_put(x, device=jax.devices()[0]), unet_params)
# text_encoder_params = jax.tree_map(lambda x: jax.device_put(x, device=jax.devices()[0]), text_encoder_params)
# vae_params = jax.tree_map(lambda x: jax.device_put(x, device=jax.devices()[0]), vae_params)



def t2i_inference(
    vae_params: params,
    unet_params: params,
    text_encoder_params: params,
    text_encoder_params_2: params,
    scheduler_params: params,
    prompt: jnp.array,
    neg_prompt: jnp.array,
    prompt_2: jnp.array,
    neg_prompt_2: jnp.array,
    seed: jax.random.KeyArray,
    width: int = 1024,
    height: int = 1024,
    cfg: float = 7.5,
    num_inference_steps: int = 30,
):

    # i think this is resolution embeddings? haven't check it yet
    def _get_add_time_ids(original_size, crops_coords_top_left, target_size, bs, dtype):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = jnp.array([add_time_ids] * bs, dtype=dtype)
        return add_time_ids

    # count the batch count
    batch_size = prompt.shape[0]

    # needs to be jax array
    cfg = jnp.array([cfg], dtype=jnp.float32)

    # concatenate prompt as 1 pass prompt
    # this is way more efficient than do 2 forward pass
    combined_prompt = jnp.concatenate([neg_prompt, prompt], axis=0)
    combined_prompt_2 = jnp.concatenate([neg_prompt_2, prompt_2], axis=0)
    # see if replicating prompt helps speed up this inference
    # combined_prompt = jax.device_put(combined_prompt, device=sharding.reshape(8,1))

    # get embeddings from the text
    encoder_text_embeddings = text_encoder(
        input_ids=combined_prompt,
        params=text_encoder_params,
        train=False,
        output_hidden_states=True
    )
    # get embeddings from the text
    encoder_text_embeddings_2 = text_encoder_2(
        input_ids=combined_prompt_2,
        params=text_encoder_params_2,
        train=False,
        output_hidden_states=True
    )
    # grab second last hidden states 
    encoder_text_embeddings = encoder_text_embeddings["hidden_states"][-2]
    pooled_text_embeddings_2 = encoder_text_embeddings_2["text_embeds"]
    encoder_text_embeddings_2 = encoder_text_embeddings_2["hidden_states"][-2]
    # make sure it's not sharded but replicated (might toggle this off for now)
    # encoder_text_embeddings = jax.lax.with_sharding_constraint(
    #     encoder_text_embeddings, shardings=sharding.replicate()
    # )

    # resolution embedding ??
    add_time_ids = _get_add_time_ids(
            (1024, 1024), (0, 0), (1024, 1024), batch_size*2, dtype=jnp.bfloat16
        )

    # concatenate both embedding as 1 and then we also need text encoder 2 embedding as separate input too
    encoder_text_embeddings = jnp.concatenate([encoder_text_embeddings,encoder_text_embeddings_2], axis=-1)
    # extra embedding from text encoder 2 and resolution ?
    added_cond_kwargs = {"text_embeds": pooled_text_embeddings_2, "time_ids": add_time_ids}
    # generate random latent
    latents_shape = (
        batch_size,
        4,
        height // 8,
        width // 8,
    )

    latents = jax.random.normal(seed, shape=latents_shape, dtype=jnp.float32)
    # see if replicating latent helps speed up this inference
    # latents = jax.device_put(latents, device=sharding.reshape(8,1,1,1))

    # scale the initial noise by the scale required by the scheduler
    latents = latents * scheduler_params.init_noise_sigma

    scheduler_state = noise_scheduler.set_timesteps(
        scheduler_params, num_inference_steps=num_inference_steps, shape=latents.shape
    )

    def single_step_pass(
        loop_counter: int, loop_state:Union[noise_scheduler_state, jnp.array]
    ):  
        scheduler_state, latents = loop_state
        # need 2 latents one for prompt and the other for neg prompt so just duplicate this
        # this is used for classifier free guidance, to contrast the vector towards bad stuff
        latents_input = jnp.concatenate([latents] * 2)

        # get scheduler timestep (reverse time step) from this loop step
        t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[loop_counter]
        # create broadcastable array for the batch dim
        timestep = jnp.broadcast_to(t, latents_input.shape[0])

        # get sample noised latent from the schedule
        latents_input = noise_scheduler.scale_model_input(
            scheduler_state, latents_input, t
        )

        # predict the noise residual to be substracted
        noise_pred = unet.apply(
            variables={"params": unet_params},
            sample=jnp.array(latents_input),  # just to make sure
            timesteps=jnp.array(timestep, dtype=jnp.int32),  # just to make sure
            encoder_hidden_states=encoder_text_embeddings,
            added_cond_kwargs=added_cond_kwargs,
        ).sample

        # perform guidance to amplify / strengthen the positive prompt
        # split the output back to neg and positive
        noise_pred_uncond, noise_prediction_text = jnp.split(noise_pred, 2, axis=0)
        # "amplified" vector noise
        noise_pred = noise_pred_uncond + cfg * (
            noise_prediction_text - noise_pred_uncond
        )

        # "subtract" the noise and return less noised sample  and the state back
        latents, scheduler_state = noise_scheduler.step(
            scheduler_state, noise_pred, t, latents
        ).to_tuple()

        return scheduler_state, latents

    # iteratively denoise the image
    # for step_count in range(num_inference_steps):
    #     latents, scheduler_state = single_step_pass(
    #         loop_counter=step_count, scheduler_state=scheduler_state, latents=latents
    #     )
    scheduler_state, latents = jax.lax.fori_loop(0, num_inference_steps, single_step_pass, (scheduler_state, latents))

    # scale and decode the image latents with vae (undoing the training scale)
    # idk why they use this constant instead of proper normalization method ¯\_(ツ)_/¯ i guess it's simpler
    latents = 1 / 0.18215 * latents
    vae_outputs = vae.apply(
        variables={"params": vae_params},
        latents=latents,
        deterministic=True,
        method=vae.decode,
    )
    return vae_outputs

jax.profiler.start_trace("./tensorboard")
t2i = jax.jit(t2i_inference)

text_input = tokenizer(
    ["4k, cat astronaut, wearing sunglasses and holding a beer can in outer space"]*1 + [""]*1,
    padding="max_length",
    max_length=77,
    truncation=True,
    return_tensors="np",
)

pos, neg = np.split(text_input.input_ids, 2, axis=0)

text_input_2 = tokenizer_2(
    ["4k, cat astronaut, wearing sunglasses and holding a beer can in outer space"]*1 + [""]*1,
    padding="max_length",
    max_length=77,
    truncation=True,
    return_tensors="np",
)

pos_2, neg_2 = np.split(text_input.input_ids, 2, axis=0)
image = t2i(
    vae_params=vae_params, 
    unet_params=unet_params, 
    text_encoder_params=text_encoder_params, 
    text_encoder_params_2=text_encoder_params_2, 
    scheduler_params=scheduler_params, 
    prompt=pos, 
    neg_prompt=neg, 
    prompt_2=pos_2, 
    neg_prompt_2=neg_2, 
    seed=jax.random.PRNGKey(42)
)

image = image.sample
image = (image / 2 + 0.5).clip(0, 1).transpose(0, 2, 3, 1)
image_np = np.array((image[0]*255).astype(jnp.uint8))
image_pil = Image.fromarray(image_np)
image_pil.save("test_non_shard.png")

jax.profiler.stop_trace()
print()
