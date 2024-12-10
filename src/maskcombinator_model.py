import jax
import jax.numpy as jnp
import genjax
from genjax import gen, Mask
from genjax import ChoiceMapBuilder as C
from genjax import ExactDensity, Pytree
from tensorflow_probability.substrates import jax as tfp
from genjax._src.core.interpreters.staging import Flag

import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial
import matplotlib.animation as animation
from distributions import *
from render import *
from config import * 
from utils import *
# from visualizer import *
tfd = tfp.distributions

def printd(val):
    jax.debug.print("{val}", val=val)

def masked_scan_combinator(step, **scan_kwargs):
    def scan_step_pre(state, flag):
        return flag, state

    def scan_step_post(_unused_args, masked_retval):
        return masked_retval.value, None

    # scan_step: (a, Bool) -> a
    scan_step = step.mask().dimap(pre=scan_step_pre, post=scan_step_post)

    return scan_step.scan(**scan_kwargs)


""" Define Distributions """
@gen
def init_firefly():
    init_x = genjax.uniform(1., SCENE_SIZE.astype(jnp.float32)) @ "x"
    init_y = genjax.uniform(1., SCENE_SIZE.astype(jnp.float32)) @ "y"

    vx = genjax.truncated_normal(0., .5, MIN_VELOCITY, MAX_VELOCITY) @ "vx"
    vy = genjax.truncated_normal(0., .5, MIN_VELOCITY, MAX_VELOCITY) @ "vy"

    blink_rate = genjax.normal(0.1, 0.01) @ "blink_rate"
    blinking = False
    state_duration = jax.lax.select(True, 0, 0)

    firefly = {
        "x": init_x,
        "y": init_y,
        "vx": vx,
        "vy": vy,
        "blink_rate": blink_rate,
        "blinking": blinking,
        "state_duration": state_duration
    }

    return firefly


@gen 
def step_firefly(firefly):
    """
    Dynamics for a single firefly

    Args:
        firefly: dictionary of firefly state
    Returns: 
        firefly: dictionary of updated firefly state
    """
    x = firefly["x"]
    y = firefly["y"]
    vx = firefly["vx"]
    vy = firefly["vy"]
    base_blink_rate = firefly["blink_rate"]
    was_blinking = firefly["blinking"]
    state_duration = firefly["state_duration"]

    # Switch direction on collision
    vx = jnp.where((x + vx > SCENE_SIZE) | (x + vx < 0.), -vx, vx)
    vy = jnp.where((y + vy > SCENE_SIZE) | (y + vy < 0.), -vy, vy)
    
    # Update position
    new_x = x + vx  
    new_y = y + vy 

    # Add some noise to position and velocity
    new_x = genjax.truncated_normal(new_x, 0.01, 0., SCENE_SIZE.astype(jnp.float32)) @ "x" 
    new_y = genjax.truncated_normal(new_y, 0.01, 0., SCENE_SIZE.astype(jnp.float32)) @ "y"
    
    new_vx = genjax.truncated_normal(vx, .3, MIN_VELOCITY, MAX_VELOCITY) @ "vx"
    new_vy = genjax.truncated_normal(vx, .3, MIN_VELOCITY, MAX_VELOCITY) @ "vy"

    # Update blinking - currently a finite state machine with weighted on/off
    # current_blink_rate = jnp.where(was_blinking, 1 / state_duration, base_blink_rate)
    blink = genjax.flip(base_blink_rate) @ "blinking"
    
    # Keep count of duration of current state or restart the counter on change
    new_state_duration = jnp.where(blink == was_blinking, state_duration + 1, 1)

    firefly = {
        "x": new_x,
        "y": new_y,
        "vx": new_vx,
        "vy": new_vy,
        "blink_rate": base_blink_rate,
        "blinking": blink,
        "state_duration": new_state_duration
    }
    
    return firefly

@gen
def get_pixel_observation(xs, ys, blinks, state_durations):
    """
    Generate deterministic rendering given xs, ys and blinks,
    and apply small amount of noise (truncated, to keep everything in nice pixel space)

    Assumes xs, ys, blinks, state_durations have been filled with 0s in place of masks
    """
    rendered = render_frame(xs, ys, blinks, state_durations)

    # Run detection
    noisy_obs = genjax.truncated_normal(rendered, 0.01, 0.0, 1.0) @ "pixels"
    return noisy_obs

@gen 
def get_observed_blinks(xs, ys, blinks):
    observed_xs = jnp.full_like(xs, -10.)
    observed_ys = jnp.full_like(ys, -10.)
    
    # Use where to conditionally select values
    observed_xs = jnp.where(blinks, xs, observed_xs)
    observed_ys = jnp.where(blinks, ys, observed_ys)
    
    observed_xs = genjax.normal(observed_xs, 0.01) @ "observed_xs"
    observed_ys = genjax.normal(observed_ys, 0.01) @ "observed_ys"
    return jnp.stack([observed_xs, observed_ys])

@gen
def step_and_observe(prev_state):
    masked_fireflies, prev_obs = prev_state
    masks = masked_fireflies.flag
    firefly_vals = masked_fireflies.value
    fireflies = step_firefly.mask().vmap(in_axes=(0, 0))(masks, firefly_vals) @ "dynamics"
    firefly_vals = fireflies.value
    xs = get_masked_values(masks,firefly_vals["x"])
    ys = get_masked_values(masks, firefly_vals["y"])
    blinks = get_masked_values(masks, firefly_vals["blinking"])
    state_durations = get_masked_values(masks, firefly_vals["state_duration"])
    #observation = get_pixel_observation(xs, ys, blinks, state_durations) @ "observations"
    observation = get_observed_blinks(xs, ys, blinks) @ "observations"
    
    return (fireflies, observation)

mask_iterate_step = genjax.masked_iterate_final()(step_and_observe)

@gen    
def multifirefly_model(max_fireflies, temporal_mask): 
    """
    Samples a number of fireflies and runs a vmapped `step_and_observe` model
    for a number of time steps, masking out unused fireflies. 

    Args:
        max_fireflies (Array[Int]): int array of indices (jnp.arange(1, max_fireflies))
        temporal_mask (Flag(Array[Bool])): Boolean array to mask timesteps (for SMC). 

    Returns:
        fireflies: dict representation of all the fireflies (including masked ones)
        observations: jnp.ndarray (t, scene_size, scene_size) of observed values
    """

    n_fireflies = labcat(unicat(max_fireflies), max_fireflies) @ "n_fireflies"
    masks = jnp.array(max_fireflies <= n_fireflies)
    init_states = init_firefly.mask().vmap(in_axes=(0))(masks) @ "init"

    #init_obs = jnp.zeros((SCENE_SIZE, SCENE_SIZE))
    init_obs = jnp.zeros((2, len(max_fireflies))).astype(jnp.float32)
    fireflies, observations = mask_iterate_step(
                            (init_states, init_obs), temporal_mask) @ "steps"
    return fireflies, observations


# def main():
#     key = jax.random.PRNGKey(3124)
#     key, subkey = jax.random.split(key)
#     max_fireflies = jnp.arange(1, 5)
#     multi_model_jit = jax.jit(multifirefly_model.importance)
#     constraints = C["n_fireflies"].set(1)
#     key, subkey = jax.random.split(key)

#     run_until = jnp.arange(TIME_STEPS) < TIME_STEPS
#     tr, weight = multi_model_jit(subkey, constraints, (max_fireflies, run_until,))
#     chm = tr.get_sample()
#     x = chm["steps", ..., "dynamics", ..., "x"]
#     print("Generating animation")
#     frames = get_frames(chm)
#     ani = animate(frames, 20)
#     # print("Saving animation...")
#     # ani.save("animations/genjax_model/maxglowsize_1.gif", writer="imagemagick", fps=20)
#     plt.show()

# if __name__ == "__main__":
#     main()