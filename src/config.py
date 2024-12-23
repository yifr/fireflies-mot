import jax.numpy as jnp

SCENE_SIZE = jnp.float32(32.)
MIN_VELOCITY = jnp.float32(-2.)
MAX_VELOCITY = jnp.float32(2.)
SCENE_Y, SCENE_X = jnp.mgrid[0:SCENE_SIZE, 0:SCENE_SIZE] 
TIME_STEPS = 30
MAX_GLOW_SIZE = jnp.float32(1.)

BLINK_MEAN = 0.4
BLINK_STD = 0.05