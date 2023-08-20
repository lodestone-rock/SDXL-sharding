#!/bin/bash
virtualenv -p python3.9 test_env
test_env/bin/python3 -m pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
test_env/bin/python3 -m pip install transformers diffusers flax optax
test_env/bin/python3 -m pip install tf-nightly tb-nightly tbp-nightly