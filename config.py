import jax.numpy as jnp

SCENE_SIZE = jnp.int16(64)
MIN_VELOCITY = -2.
MAX_VELOCITY = 2.
SCENE_Y, SCENE_X = jnp.mgrid[0:SCENE_SIZE, 0:SCENE_SIZE] 
TIME_STEPS = 100