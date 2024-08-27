import jax.numpy as jnp
import jax
import genjax
from genjax import gen
from config import *

@jax.jit
def calculate_glow(coordinate, glow_size,):
    distances = jnp.sqrt((SCENE_X - jnp.round(coordinate[1]))**2 + (SCENE_Y - jnp.round(coordinate[0]))**2)
    glow_values = jnp.exp(-(distances**2) / (2 * glow_size**2))
    return glow_values

@jax.jit
def render_frame(xs, ys, blinks, state_durations, max_glow_size):
    # Extract x, y coordinates and blinking values for the current frame
    coordinates = jnp.stack([xs, ys], axis=-1)
    blink_durations = state_durations * blinks
    glow_sizes = jnp.where(blink_durations < max_glow_size, blink_durations, max_glow_size)
    glow_values = jax.vmap(calculate_glow, in_axes=(0, 0))(coordinates, glow_sizes)
    glow_grid = jnp.zeros((SCENE_SIZE, SCENE_SIZE)) 

    def accumulate_glow(carry, values):
        glow_grid = jnp.where(values > 0.01, values + carry, carry)
        return glow_grid, None

    # Scan across fireflies to create a composite image with additive glow
    final_glow_grid = jax.lax.scan(accumulate_glow, glow_grid, glow_values)[0]
    
    return final_glow_grid

@gen
def observe_fireflies(xs, ys, blinks, state_durations, max_glow_size=2.):
    v_render_frame = jax.vmap(render_frame, in_axes=(0, 0, 0, 0, None))
    rendered = v_render_frame(xs, ys, blinks, state_durations, max_glow_size)
    noisy_obs = genjax.truncated_normal(rendered, 0.01, 0.0, 1.0) @ "pixels"
    return noisy_obs