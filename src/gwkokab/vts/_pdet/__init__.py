import gwpopulation


# We only support JAX backend
# https://github.com/ColmTalbot/wcosmo/issues/3#issuecomment-2426836308
gwpopulation.set_backend("jax")


from .O3 import pdet_O3 as pdet_O3
