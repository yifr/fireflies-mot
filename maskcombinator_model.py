import jax
import jax.numpy as jnp
import genjax
from genjax import gen
from genjax import ChoiceMapBuilder as C
from genjax import ExactDensity, Pytree
from tensorflow_probability.substrates import jax as tfp
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial
import matplotlib.animation as animation
from distributions import *
from render import *
from config import * 
tfd = tfp.distributions

def masked_iterate_combinator(step, **scan_kwargs):
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

    vx = genjax.truncated_normal(0., 1., MIN_VELOCITY, MAX_VELOCITY) @ "vx"
    vy = genjax.truncated_normal(0., 1., MIN_VELOCITY, MAX_VELOCITY) @ "vy"

    blink_rate = genjax.normal(0.1, 0.01) @ "blink_rate"
    blinking = jax.lax.select(True, 0, 0)
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
    Scanned function that accepts prev state and 
    variable that stores full history of values

    Args:
    firefly: dictionary of firefly state
    _: list of previous states

    Returns: (when scanned)
    firefly: dictionary of updated firefly state
    history: list of updated states
    """
    x = firefly["x"]
    y = firefly["y"]
    vx = firefly["vx"]
    vy = firefly["vy"]
    blink_rate = firefly["blink_rate"]
    blinking = firefly["blinking"]
    state_duration = firefly["state_duration"]

    # Switch on collision
    vx = jnp.where((x + vx > SCENE_SIZE) | (x + vx < 0.), -vx, vx)
    vy = jnp.where((y + vy > SCENE_SIZE) | (y + vy < 0.), -vy, vy)
    
    new_x = x + vx  
    new_y = y + vy 

    new_x = genjax.truncated_normal(new_x, 0.1, 0., SCENE_SIZE.astype(jnp.float32)) @ "x" 
    new_y = genjax.truncated_normal(new_y, 0.1, 0., SCENE_SIZE.astype(jnp.float32)) @ "y"
    
    new_vx = genjax.truncated_normal(vx, .5, MIN_VELOCITY, MAX_VELOCITY) @ "vx"
    new_vy = genjax.truncated_normal(vx, .5, MIN_VELOCITY, MAX_VELOCITY) @ "vy"

    # Update blinking - currently a finite state machine with weighted on/off
    current_blink_rate = jnp.where(blinking, 1 - blink_rate, blink_rate)
    blink = genjax.flip(current_blink_rate) @ "blink"
    
    # Keep count of duration of current state
    state_change = jnp.where(blink == blinking, 0, 1)
    new_state_duration = jnp.where(state_change, 1, state_duration + 1)

    firefly = {
        "x": new_x,
        "y": new_y,
        "vx": new_vx,
        "vy": new_vy,
        "blink_rate": blink_rate,
        "blinking": blink,
        "state_duration": new_state_duration
    }
    
    return firefly

@gen
def single_firefly_model():
    firefly = init_firefly() @ "init"
    firefly, chain = step_firefly(firefly, []) @ "dynamics"
    return (firefly, chain)


def get_masked_values(values, mask, fill_value=0.):
    mask = jnp.expand_dims(mask, axis=-1)
    return jnp.where(mask, values, fill_value)

@gen
def step_and_observe(carry):
    masked_fireflies, max_glow_size, prev_observation = carry
    masks = masked_fireflies.flag
    firefly_vals = masked_fireflies.value
    step_fn = jax.vmap(step_firefly.mask(), in_axes=(0, None))
    fireflies = step_fn(masks, firefly_vals) @ "dynamics"  
    firefly_vals = fireflies.value
    jax.debug.print("{f}", f=fireflies)
    xs = get_masked_values(firefly_vals["x"], masks)
    ys = get_masked_values(firefly_vals["y"], masks)
    blinks = get_masked_values(firefly_vals["blinking"], masks)
    state_durations = get_masked_values(firefly_vals["state_duration"], masks)
    observation = observe_fireflies(xs, ys, blinks, state_durations, max_glow_size) @ "observations"
    return (fireflies, max_glow_size, observation)

@gen 
def multifirefly_model(max_fireflies, max_glow_size): 
    n_fireflies = labcat(unicat(max_fireflies), max_fireflies) @ "n_fireflies"
    masks = max_fireflies <= n_fireflies
    init_states = jax.vmap(init_firefly.mask(), in_axes=(0))(masks) @ "init"
    init_obs = jnp.zeros((SCENE_SIZE, SCENE_SIZE))
    jax.debug.print("{i}", i=init_states.value)
    temporal_mask = jnp.arange(TIME_STEPS) < TIME_STEPS
    fireflies, observations = masked_iterate_combinator(step_and_observe, n=TIME_STEPS)((init_states, max_glow_size, init_obs), temporal_mask) @ "steps"
    # vmask_model = single_firefly_model.mask().vmap(in_axes=(0))
    # fireflies = vmask_model(masks) @ "fireflies"

    # chain = fireflies.value[1]
    # xs = get_masked_values(chain["x"], masks)
    # ys = get_masked_values(chain["y"], masks)
    # blinks = get_masked_values(chain["blinking"], masks)
    # state_durations = get_masked_values(chain["state_duration"], masks)

    # observations = observe_fireflies(xs, ys, blinks, state_durations, max_glow_size) @ "observations"
    return fireflies, observations

def get_frames(chm):
    observations = chm["observations", "pixels"]
    frames = []
    for i in range(observations.shape[0]):
        frames.append(observations[i])
    return frames

def animate(frames, fps):
    fig, ax = plt.subplots()
    img = ax.imshow(frames[0], vmin=0, vmax=1, cmap="hot")  

    def update(frame):
        img.set_data(frames[frame])  
        ax.set_title(f"Frame {frame}")
        return [img]  

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000/fps, blit=True)
    return ani


def main():
    key = jax.random.PRNGKey(3124)
    key, subkey = jax.random.split(key)
    max_fireflies = jnp.arange(1, 5)
    max_glow_size = 1.4
    steps = jnp.arange(10)

    multi_model_jit = jax.jit(multifirefly_model.simulate)
    chm = C["n_fireflies"].set(3)

    tr = multi_model_jit(subkey, (max_fireflies, max_glow_size,))

    key, subkey = jax.random.split(key)
    tr = multi_model_jit(subkey,(max_fireflies, max_glow_size,))

    chm = tr.get_sample()
    print(chm)
    print("Generating animation")
    frames = get_frames(chm)
    ani = animate(frames, 20)
    print("Saving animation...")
    # ani.save("animations/genjax_model/maxglowsize_1.gif", writer="imagemagick", fps=20)
    plt.show()

if __name__ == "__main__":
    main()