
import json
import ast

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from typing import Union
from PIL import Image
import optax
from flax.training import train_state

from diffusers import (
    FlaxAutoencoderKL,
    FlaxDDPMScheduler,
    FlaxStableDiffusionPipeline,
    # FlaxUNet2DConditionModel,
)
from models import FlaxUNet2DConditionModel
from transformers import CLIPFeatureExtractor, CLIPTokenizer, FlaxCLIPTextModel

import diffusers.schedulers.scheduling_ddim_flax

from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

# global var
adam_to_lion_scale_factor = 7
u_net_learning_rate = 1e-5
text_encoder_learning_rate = 1e-5


# typehint definition
transformed_params = dict
params = dict
rng = jax.random.PRNGKey
noise_scheduler_state = diffusers.schedulers.scheduling_ddim_flax.DDIMSchedulerState

sharding = PositionalSharding(mesh_utils.create_device_mesh((jax.device_count(),)))
use_offset_noise = False
strip_bos_eos_token = True


def read_json_file(file_path):
    try:
        with open(file_path, 'r') as json_file:
            data_dict = json.load(json_file)
        return data_dict
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file: {file_path}")
        return None

def save_model_tree_as_json(file_name: str, params: dict) -> None:

    # Save the dictionary as JSON
    with open(file_name, "w") as json_file:
        json.dump(params, json_file)


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


# convert it as actual sharding 
def predefined_sharding(layouts:dict) -> dict:
    # convert list as sharding and none as replicate
    def _convert(param):
        if param != None:
            param = sharding.reshape(param)
        else:
            param = sharding.replicate() 
        return param
    layouts = jax.tree_map(lambda x: _convert(ast.literal_eval(x)), layouts)
    return layouts


def all_same_bool_values(d):
    values = set()
    for v in d.values():
        if isinstance(v, bool):
            values.add(v)
        elif isinstance(v, dict):
            values.update(all_same_bool_values(v))
    return values



model_dir = "/home/teor/secondary_storage/tpu8/model/fluffyrock-576-704-832-960-1088-lion-e130"
# load the model params and model object

tokenizer = CLIPTokenizer.from_pretrained(model_dir, subfolder="tokenizer")

unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
    model_dir, subfolder="unet", dtype=jnp.bfloat16, use_memory_efficient=True
)

text_encoder, text_encoder_params = FlaxCLIPTextModel.from_pretrained(
    model_dir, subfolder="text_encoder", dtype=jnp.bfloat16, _do_init=False
)

vae, vae_params = FlaxAutoencoderKL.from_pretrained(
    model_dir,
    dtype=jnp.bfloat16,
    subfolder="vae",
)

noise_scheduler = FlaxDDPMScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000,
    prediction_type="v_prediction",
)

noise_scheduler_state = noise_scheduler.create_state()

# shard the weights across device
# unet_params = jax.tree_map(shard_weight_column, unet_params)
# text_encoder_params = jax.tree_map(shard_weight_column, text_encoder_params)
# vae_params = jax.tree_map(shard_weight_column, vae_params)

# grab tree sharding layout for each layer
unet_params_shard_layout = read_json_file("unet_state_layout.json")
text_encoder_shard_layout = read_json_file("clip_sharding_layout.json")
vae_params_shard_layout = read_json_file("vae_state_layout.json")

unet_params_shard_layout = predefined_sharding(unet_params_shard_layout)
text_encoder_shard_layout = predefined_sharding(text_encoder_shard_layout)
vae_params_shard_layout = predefined_sharding(vae_params_shard_layout)

# jax.profiler.start_trace("./tensorboard")

unet_params = jax.tree_map(lambda params, layout: jax.device_put(params, device=layout), unet_params, unet_params_shard_layout)
text_encoder_params = jax.tree_map(lambda params, layout: jax.device_put(params, device=layout), text_encoder_params, text_encoder_shard_layout)
vae_params = jax.tree_map(lambda params, layout: jax.device_put(params, device=layout), vae_params, vae_params_shard_layout)

u_net_constant_scheduler = optax.constant_schedule(
    u_net_learning_rate / adam_to_lion_scale_factor
)
text_encoder_constant_scheduler = optax.constant_schedule(
    text_encoder_learning_rate / adam_to_lion_scale_factor
)

# optimizer for U-Net
u_net_lion = optax.lion(
    learning_rate=u_net_constant_scheduler,
    b1=0.9,
    b2=0.99,
    weight_decay=1e-2 * adam_to_lion_scale_factor,
)
u_net_optimizer = optax.chain(
    optax.clip_by_global_norm(1),  # prevent explosion
    u_net_lion,
)

# optimizer for CLIP text encoder
text_encoder_lion = optax.lion(
    learning_rate=text_encoder_constant_scheduler,
    b1=0.9,
    b2=0.99,
    weight_decay=1e-2 * adam_to_lion_scale_factor,
)
text_encoder_optimizer = optax.chain(
    optax.clip_by_global_norm(1),  # prevent explosion
    text_encoder_lion,
)


unet_state = train_state.TrainState.create(
    apply_fn=unet.__call__,
    params=unet_params,
    tx=u_net_optimizer
)
del unet_params

text_encoder_state = train_state.TrainState.create(
    apply_fn=text_encoder.__call__,
    params=text_encoder_params,
    tx=text_encoder_optimizer
)
del text_encoder_params

def train_step(unet_state, text_encoder_state, vae_params, batch, train_rng:jax.random.PRNGKey):
    # generate rng and return new_train_rng to be used for the next iteration step
    # rng is comunicated though device aparently
    dropout_rng, sample_rng, new_train_rng = jax.random.split(
        train_rng, num=3)


    # trainable params is passed as an argument while 
    # non trainable params are implicitly referenced in loss calculation
    params = {
        "text_encoder": text_encoder_state.params,
        "unet": unet_state.params
    }

    def compute_loss(params):
        # Convert images to latent space
        vae_outputs = vae.apply(
            {"params": vae_params},
            batch["pixel_values"],
            deterministic=True,
            method=vae.encode
        )

        # get sample distribution from VAE latent
        latents = vae_outputs.latent_dist.sample(sample_rng)
        # (NHWC) -> (NCHW)
        latents = jnp.transpose(latents, (0, 3, 1, 2))
        # weird scaling don't touch it's a lazy normalization
        latents = latents * 0.18215

        # Sample noise that we'll add to the latents
        # I think I should combine this with the first noise seed generator
        noise_offset_rng, noise_rng, timestep_rng = jax.random.split(
            sample_rng, num=3)
        noise = jax.random.normal(noise_rng, latents.shape)
        if use_offset_noise:
            # mean offset noise, why add offset?
            # here https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise_offset = jax.random.normal(
                noise_offset_rng,
                (latents.shape[0], latents.shape[1], 1, 1)
            ) * 0.1
            noise = noise + noise_offset

        # Sample a random timestep for each image
        bsz = latents.shape[0]
        timesteps = jax.random.randint(
            timestep_rng,
            (bsz,),
            0,
            noise_scheduler.config.num_train_timesteps,
        )

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(
            noise_scheduler_state,
            latents,
            noise,
            timesteps
        )
        print(batch["input_ids"].shape)
        encoder_hidden_states = text_encoder_state.apply_fn(
            batch["input_ids"],
            params=params["text_encoder"],
            dropout_rng=dropout_rng,
            train=True
        )[0]
        print(encoder_hidden_states.shape)
        # reshape encoder_hidden_states to shape (batch, token_append, token, hidden_states)
        encoder_hidden_states = jnp.reshape(
            encoder_hidden_states,
            (latents.shape[0], -1, 77, encoder_hidden_states.shape[-1]),
        )
        print(encoder_hidden_states.shape)

        if strip_bos_eos_token:
            encoder_hidden_states = jnp.concatenate(
                [
                    # first encoder hidden states without eos token
                    encoder_hidden_states[:, 0, :-1, :],
                    # the rest of encoder hidden states without both bos and eos token
                    jnp.reshape(
                        encoder_hidden_states[:, 1:-1, 1:-1, :],
                        (
                            encoder_hidden_states.shape[0],
                            -1,
                            encoder_hidden_states.shape[-1]
                        )
                    ),
                    # last encoder hidden states without bos token
                    encoder_hidden_states[:, -1, 1:, :]
                ],
                axis=1
            )
        else:
            # reshape encoder_hidden_states to shape (batch, token_append & token, hidden_states)
            encoder_hidden_states = jnp.reshape(
                encoder_hidden_states,
                (encoder_hidden_states.shape[0], -
                    1, encoder_hidden_states.shape[-1])
            )
        print(encoder_hidden_states.shape)

        # Predict the noise residual because predicting image is hard :P
        # essentially try to undo the noise process
        model_pred = unet.apply(
            {"params": params["unet"]},
            noisy_latents,
            timesteps,
            encoder_hidden_states,
            train=True
        ).sample

        # Get the target for loss depending on the prediction type
        # sd1.x use epsilon aka noise residual but sd2.1 use velocity prediction
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(
                noise_scheduler_state,
                latents,
                noise,
                timesteps
            )
        else:
            # panic!!
            raise ValueError(
                f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        # MSE loss
        loss = (target - model_pred) ** 2
        loss = loss.mean()

        return loss

    # perform autograd
    grad_fn = jax.value_and_grad(compute_loss)
    loss, grad = grad_fn(params)

    # update weight and bias value
    new_unet_state = unet_state.apply_gradients(grads=grad["unet"])
    new_text_encoder_state = text_encoder_state.apply_gradients(
        grads=grad["text_encoder"])

    # calculate loss
    metrics = {"loss": loss}

    return new_unet_state, new_text_encoder_state, metrics, new_train_rng

# ===============[compile to device]=============== #


jax.profiler.start_trace("./tensorboard")
p_train_step = jax.jit(train_step)# , donate_argnums=(0, 1))


train_rngs = rng(2)
# dummy batch input
current_batch = {
    'attention_mask': jnp.arange(1 * 1 * 3 * 77).reshape(1 * 1, 3, 77), 
    'input_ids': jnp.arange(1 * 3 * 77).reshape(1 * 3, 77), 
    'pixel_values': jax.random.uniform(train_rngs, shape=(1 * 1, 3, 1088, 1088))
}

batch = jax.tree_map(
    lambda x: jax.device_put(x, device=sharding.replicate()), current_batch
)

unet_state, text_encoder_state, train_metric, train_rngs = p_train_step(
    unet_state,
    text_encoder_state,
    vae_params,
    batch,
    train_rngs
)
jax.profiler.stop_trace()

print()