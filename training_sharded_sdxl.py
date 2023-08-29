
import json
import ast
import gc

import jax
import re
import jaxlib
import jax.numpy as jnp
import numpy as np
from PIL import Image
from typing import Union, Any
from PIL import Image
import optax
from flax.training import train_state
import flax
import partition_pattern

from diffusers import (
    FlaxAutoencoderKL,
    FlaxDDPMScheduler,
    FlaxStableDiffusionPipeline,
    # FlaxUNet2DConditionModel,
)
from models import FlaxUNet2DConditionModel, FlaxCLIPTextModel, FlaxCLIPTextModelWithProjection
from transformers import CLIPTokenizer

import diffusers.schedulers.scheduling_ddim_flax

from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

from jax.sharding import Mesh
from jax.sharding import PartitionSpec
from jax.sharding import NamedSharding
from jax.experimental import mesh_utils

P = PartitionSpec
# adjust this sharding mesh to create appropriate sharding rule
# assume we have 8 device
# (1,8) = model parallel
# (8,1) = data parallel
# (4,2)/(2,4) = model data parallel
devices = mesh_utils.create_device_mesh((1,8))
mesh = Mesh(devices, axis_names=('dp', 'mp')) 


# global var
adam_to_lion_scale_factor = 7
u_net_learning_rate = 1e-5
text_encoder_learning_rate = 1e-5


# typehint definition
transformed_params = dict
params = dict
rng = jax.random.PRNGKey
noise_scheduler_state = diffusers.schedulers.scheduling_ddim_flax.DDIMSchedulerState
leaf = Any
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

def create_flattened_tree_json(params:dict, name:str) -> None:
    flattened_tree = flax.traverse_util.flatten_dict(params)
    flattened_tree = jax.tree_map(shard_weight_column, flattened_tree)
    flattened_tree_shape = jax.tree_map(lambda x: repr(x.shape), flattened_tree)
    flattened_tree = jax.tree_map(lambda x: repr(x.sharding), flattened_tree)
    flattened_tree = {".".join(key): values for key,values in flattened_tree.items()}
    flattened_tree_shape = {".".join(key): values for key,values in flattened_tree_shape.items()}
    save_model_tree_as_json(f"flattened_{name}_sharding.json", flattened_tree)
    save_model_tree_as_json(f"flattened_{name}_shape.json", flattened_tree_shape)

def debug_tree_json(params:dict, name:str) -> None:
    flattened_tree = flax.traverse_util.flatten_dict(params)
    flattened_tree_shape = jax.tree_map(lambda x: repr(x.shape), flattened_tree)
    flattened_tree = jax.tree_map(lambda x: repr(x.sharding), flattened_tree)
    flattened_tree = {".".join(key): values for key,values in flattened_tree.items()}
    flattened_tree_shape = {".".join(key): values for key,values in flattened_tree_shape.items()}
    save_model_tree_as_json(f"flattened_{name}_sharding.json", flattened_tree)
    save_model_tree_as_json(f"flattened_{name}_shape.json", flattened_tree_shape)

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

# convert it as actual sharding 
def predefined_mesh_sharding(layouts:dict) -> dict:
    # convert string like (None, 'mp') to actual tuple and turn it to sharding rule
    def _convert(param):
        param = NamedSharding(mesh, P(*param))
        return param
    layouts = jax.tree_map(lambda x: _convert(ast.literal_eval(x)), layouts)
    return layouts

def shard_remainder_state_param(param_leaf):
    # if it already sharded then ignore it
    if hasattr(param_leaf, "sharding"):
        # if it's not sharded then shard it
        if type(param_leaf.sharding) == jaxlib.xla_extension.SingleDeviceSharding:
            shard_rule = NamedSharding(mesh, P())
        else:
            shard_rule = param_leaf.sharding
    # shard / replicate pesky remainder params
    else:
        shard_rule = NamedSharding(mesh, P())
    return shard_rule


def all_same_bool_values(d):
    values = set()
    for v in d.values():
        if isinstance(v, bool):
            values.add(v)
        elif isinstance(v, dict):
            values.update(all_same_bool_values(v))
    return values


# create a parameter path when i call this function using tree_map_with_path
# usefull to define sharding behaviour for each param in the param tree or flax train state
# stolen from EasyLM utils :P
def tree_path_to_string(path:str, sep:str="."):
    keys = []
    for key in path:
        if isinstance(key, jax.tree_util.SequenceKey):
            keys.append(str(key.idx))
        elif isinstance(key, jax.tree_util.DictKey):
            keys.append(str(key.key))
        elif isinstance(key, jax.tree_util.GetAttrKey):
            keys.append(str(key.name))
        elif isinstance(key, jax.tree_util.FlattenedIndexKey):
            keys.append(str(key.key))
        else:
            keys.append(str(key))
    if sep is None:
        return tuple(keys)
    return sep.join(keys)

# regex match if the pattern exist from json 
# used for applying sharding layour for each layer
def shard_based_on_lut(lookup_list: list, param:tuple, sep:str=".") -> leaf:

    param_path=tree_path_to_string(param[0], sep)
    # param should be (tree_path, param_value)
    leaf = None
    for layer_path, sharding_layout in lookup_list:
        # check if the model layer path match 
        # then shard accodring to the lookup table
        match = bool(re.search(layer_path, param_path))
        if match:
            leaf = jax.device_put(param[1], device=NamedSharding(mesh, P(*sharding_layout)))
            break
    # if not found just replicate it
    # here it assumes that named sharding is already defined
    if leaf == None:
        print(param_path)
        leaf = jax.device_put(param[1], device=NamedSharding(mesh, P()))
    return leaf

model_dir = "/home/teor/secondary_storage/SDXL-sharding/stable-diffusion-xl-base-1.0-flax"
# load the model params and model object

tokenizer = CLIPTokenizer.from_pretrained(model_dir, subfolder="tokenizer")
tokenizer_2 = CLIPTokenizer.from_pretrained(model_dir, subfolder="tokenizer_2")

unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
    model_dir, subfolder="unet", dtype=jnp.bfloat16, use_memory_efficient=False
)

text_encoder, text_encoder_params = FlaxCLIPTextModel.from_pretrained(
    model_dir, subfolder="text_encoder", dtype=jnp.bfloat16, _do_init=False
)

text_encoder_2, text_encoder_2_params = FlaxCLIPTextModelWithProjection.from_pretrained(
    model_dir, subfolder="text_encoder_2", dtype=jnp.bfloat16, _do_init=False
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
    # prediction_type="v_prediction",
)

noise_scheduler_state = noise_scheduler.create_state()


# jax.profiler.start_trace("./tensorboard")


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

text_encoder_state = train_state.TrainState.create(
    apply_fn=text_encoder.__call__,
    params=text_encoder_params,
    tx=text_encoder_optimizer
)

# trained
unet_state = jax.tree_util.tree_map_with_path(lambda path, leaf: shard_based_on_lut(partition_pattern.unet_partition, (path, leaf)),unet_state)
text_encoder_state = jax.tree_util.tree_map_with_path(lambda path, leaf: shard_based_on_lut(partition_pattern.clip_partition, (path, leaf)),text_encoder_state)
# not trained
text_encoder_2_params = jax.tree_util.tree_map_with_path(lambda path, leaf: shard_based_on_lut(partition_pattern.clip_partition, (path, leaf)),text_encoder_2_params)
# vae_params = jax.tree_util.tree_map_with_path(lambda path, leaf: shard_based_on_lut(vae_params_shard_layout, (path, leaf)),vae_params)
# replicate vae for now, i think xla is gud at merging this shitton conv ops
vae_params = jax.tree_util.tree_map(lambda leaf: jax.device_put(leaf, device=NamedSharding(mesh, P())),vae_params)
# cast as bf16 since it's not trained
text_encoder_2_params = jax.tree_map(lambda leaf: leaf.astype(jnp.bfloat16),text_encoder_2_params)
vae_params = jax.tree_map(lambda leaf: leaf.astype(jnp.bfloat16),vae_params)
# delete previous params because state creates a copy of it and occupy a memory
# del unet_params
# del text_encoder_params
# gc.collect()

def train_step(unet_state, text_encoder_state, text_encoder_2_params, vae_params, batch, train_rng:jax.random.PRNGKey):
    # generate rng and return new_train_rng to be used for the next iteration step
    # rng is comunicated though device aparently
    dropout_rng, sample_rng, new_train_rng = jax.random.split(
        train_rng, num=3)


    # trainable params 
    params = {
        "text_encoder": text_encoder_state.params,
        "unet": unet_state.params
    }
    # frozen params put in dict so it's clear and not implicitly referenced
    frozen_params = {
        "text_encoder_2": text_encoder_2_params,
        "vae": vae_params
    }
    # i set autograd only calculate the first params in this case so the second params is frozen
    def compute_loss(params, frozen_params):
        ### Convert images to latent space and noise the heck out of it ###
        vae_outputs = vae.apply(
            {"params": frozen_params["vae"]},
            batch["pixel_values"],
            deterministic=True,
            method=vae.encode
        )

        # get sample distribution from VAE latent
        latents = vae_outputs.latent_dist.sample(sample_rng)
        # (NHWC) -> (NCHW)
        latents = jnp.transpose(latents, (0, 3, 1, 2))
        #scaling factor is in the config so grab from that (still lazy norm thinggy)
        latents = latents * vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        # I think I should combine this with the first noise seed generator
        noise_offset_rng, noise_rng, timestep_rng = jax.random.split(
            sample_rng, num=3)
        noise = jax.random.normal(noise_rng, latents.shape)
        if use_offset_noise: # unsused
            # mean offset noise, why add offset?
            # here https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise_offset = jax.random.normal(
                noise_offset_rng,
                (latents.shape[0], latents.shape[1], 1, 1)
            ) * 0.1
            noise = noise + noise_offset

        # Sample a random timestep for each image
        batch_size = latents.shape[0]
        timesteps = jax.random.randint(
            timestep_rng,
            (batch_size,),
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

        ### text embedding guidance ###
        def _concatenate_text_encoder_latent(
            encoder_hidden_states:jnp.array, 
            batch_size:int=batch_size,
            strip_bos_eos_token:bool=strip_bos_eos_token
        ):
            # i put batch size value as default :P
            # this is a hacky way from automatic1111 to extend context length and usefull during training
            # since the context length is not easily extendable due to trained positional embedding
            # reshape encoder_hidden_states to shape (batch, token_append, token, hidden_states)
            encoder_hidden_states = jnp.reshape(
                encoder_hidden_states,
                (
                    batch_size, 
                    -1, 
                    77, 
                    encoder_hidden_states.shape[-1]
                ),
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
            return encoder_hidden_states
        


        # print(batch["input_ids"].shape)
        # train the small text encoder
        encoder_text_embeddings = text_encoder_state.apply_fn(
            input_ids=batch["input_ids_text_encoder_1"],
            params=params["text_encoder"],
            dropout_rng=dropout_rng,
            train=True,
            output_hidden_states=True
        )

        # large text encoder is frozen because i want to preserve the prior knowledge
        # this chonky text encoder is good at natural language prompt
        encoder_text_embeddings_2 = text_encoder_2(
            input_ids=batch["input_ids_text_encoder_2"],
            params=frozen_params["text_encoder_2"],
            train=False,
            output_hidden_states=True
        )
        # print(encoder_hidden_states.shape)

        # grab second last hidden states from text embedding 1
        encoder_text_embeddings = encoder_text_embeddings["hidden_states"][-2]
        # concatenate procedure, automatic1111 way
        encoder_text_embeddings  = _concatenate_text_encoder_latent(encoder_text_embeddings)
        # grab MAE pooled embedding from text embedding 2 (used for aditional guidance alongside crop res guidance)
        # since i use auto1111 guidane i just gonna average the pooled embeddings, ideally i modify the pooling before it got projected
        # TODO: modify the embedding pooling instead of doing this weird weighted average
        pooled_text_embeddings_2 = encoder_text_embeddings_2["text_embeds"].mean(axis=0)
        pooled_text_embeddings_2 = jnp.expand_dims(pooled_text_embeddings_2, axis=0)
        print(pooled_text_embeddings_2.shape)
        # grab second last hidden states from text embedding 2
        encoder_text_embeddings_2 = encoder_text_embeddings_2["hidden_states"][-2]
        # concatenate procedure, automatic1111 way
        encoder_text_embeddings_2  = _concatenate_text_encoder_latent(encoder_text_embeddings_2)

        # combined embeddings is concatenated along hidden dimension axis so it has (sequence, 2048 hidden dim)
        encoder_text_embeddings = jnp.concatenate([encoder_text_embeddings,encoder_text_embeddings_2], axis=-1)
        
        # resolution embeddings 
        def _get_res_cond_to_time_proj(original_size, crops_coords_top_left, target_size, bs, dtype):
            # original_size (h,w) crops_coords_top_left(t,l) target_size(h,w)
            res_cond_to_time_proj = list(original_size + crops_coords_top_left + target_size)
            res_cond_to_time_proj = jnp.array([res_cond_to_time_proj] * bs, dtype=dtype)
            return res_cond_to_time_proj

        # i wont do any kind of data anotation here just gonna grab the original res as guidance for now
        res_cond_to_time_proj = _get_res_cond_to_time_proj(
                original_size=(batch["pixel_values"].shape[2], batch["pixel_values"].shape[3]), # (height, width),
                crops_coords_top_left=(0, 0), 
                target_size=(batch["pixel_values"].shape[2], batch["pixel_values"].shape[3]), # (height, width),
                bs=batch_size,# *2, # i forgot why i multiply this by 2 i guess it's due to 2 text embedding and i have to do double guidance
                dtype=jnp.bfloat16 # should i use fp32 here hmmm, mebbe not it's not acumulating anything
        )
        
        # additional guidance from pooling and resolution to be projected as time embedding 
        added_cond_kwargs = {"text_embeds": pooled_text_embeddings_2, "time_ids": res_cond_to_time_proj}
        # Predict the noise residual because predicting image is hard :P
        # essentially try to undo the noise process
        model_pred = unet.apply(
            variables={"params": params["unet"]},
            sample=noisy_latents,
            timesteps=timesteps,
            encoder_hidden_states=encoder_text_embeddings,
            train=True,
            added_cond_kwargs=added_cond_kwargs,
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
    loss, grad = grad_fn(params, frozen_params)

    # update weight and bias value
    new_unet_state = unet_state.apply_gradients(grads=grad["unet"])
    new_text_encoder_state = text_encoder_state.apply_gradients(
        grads=grad["text_encoder"])

    # calculate loss
    metrics = {"loss": loss}

    return new_unet_state, new_text_encoder_state, metrics, new_train_rng # 

# ===============[compile to device]=============== #


# jax.profiler.start_trace("./tensorboard")

train_rngs = rng(2)
# dummy batch input
current_batch = {
    'attention_mask': jnp.arange(1 * 1 * 3 * 77).reshape(1 * 1, 3, 77), 
    'input_ids_text_encoder_1': jnp.arange(1 * 3 * 77).reshape(1 * 3, 77), 
    'input_ids_text_encoder_2': jnp.arange(1 * 3 * 77).reshape(1 * 3, 77), 
    'pixel_values': jax.random.uniform(train_rngs, shape=(1 * 1, 3, 1024, 1024))
}
# current_batch_shard_layout = {
#     'attention_mask': sharding.replicate(), 
#     'input_ids': sharding.replicate(), 
#     'pixel_values': sharding.replicate()
# }
current_batch_shard_layout = {
    'attention_mask': NamedSharding(mesh, P()), 
    'input_ids_text_encoder_1': NamedSharding(mesh, P()), 
    'input_ids_text_encoder_2': NamedSharding(mesh, P()), 
    'pixel_values': NamedSharding(mesh, P()),
}


p_train_step = jax.jit(
    train_step , 
    donate_argnums=(0, 1), 
    in_shardings=(
        jax.tree_map(lambda x: shard_remainder_state_param(x), unet_state),
        jax.tree_map(lambda x: shard_remainder_state_param(x), text_encoder_state),
        jax.tree_map(lambda x: shard_remainder_state_param(x), text_encoder_2_params),
        jax.tree_map(lambda x: shard_remainder_state_param(x), vae_params),
        current_batch_shard_layout,
        NamedSharding(mesh, P()),# sharding.replicate()
    ),
    out_shardings=(
        jax.tree_map(lambda x: shard_remainder_state_param(x), unet_state),
        jax.tree_map(lambda x: shard_remainder_state_param(x), text_encoder_state),
        {"loss":  NamedSharding(mesh, P())},
        NamedSharding(mesh, P()), # sharding.replicate() # not sure about this 
    )
)



batch = jax.tree_map(
    lambda x: jax.device_put(x, device=sharding.replicate()), current_batch
)

unet_state, text_encoder_state, metrics, train_rngs = p_train_step(
    unet_state,
    text_encoder_state,
    text_encoder_2_params,
    vae_params,
    batch,
    train_rngs
)

# jax.profiler.stop_trace()


# import time
# for x in range(100):
#     start = time.time()
#     unet_state, text_encoder_state, metrics, train_rngs = p_train_step(
#         unet_state,
#         text_encoder_state,
#         text_encoder_2_params,
#         vae_params,
#         batch,
#         train_rngs
#     )
#     stop = time.time()
#     print(metrics, stop-start)
print()