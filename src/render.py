import jax.numpy as jnp
import jax
import genjax
from genjax import gen
from config import *

@jax.jit
def gaussian_2d(loc, scale):
    """
    Compute a 2D gaussian on a grid, with a center and size specified by loc, scale
    """
    x, y = loc
    distances = jnp.sqrt((SCENE_X - jnp.round(x))**2 + (SCENE_Y - jnp.round(y))**2)
    values = jnp.exp(-(distances**2) / (2 * scale ** 2))
    return values

@jax.jit
def render_frame(xs, ys, blinks, state_durations):
    coordinates = jnp.stack([xs, ys], axis=-1)  # Get locations for each fireflies
    blink_durations = state_durations * blinks  # Determine if / how long they've been blinking for
    glow_sizes = jnp.where(blink_durations < MAX_GLOW_SIZE, blink_durations, MAX_GLOW_SIZE) # Cap their glow sizes
    glow_values = jax.vmap(gaussian_2d, in_axes=(0, 0))(coordinates, glow_sizes) # Draw each firefly
    final_glow_grid = jnp.sum(jnp.where(glow_values > 0.1, glow_values, 0), axis=0) # Combine the frames
    
    return final_glow_grid