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
import copy
tfd = tfp.distributions


""" Constant values """
TIME_STEPS = 100
SCENE_SIZE = jnp.int32(128)
SCENE_Y, SCENE_X = jnp.mgrid[0:SCENE_SIZE, 0:SCENE_SIZE] # Create the scene grid ahead of time to avoid jax tracer errors

""" Define Distributions """
@gen
def init_firefly():
    init_x = genjax.uniform(1., SCENE_SIZE.astype(jnp.float32)) @ "x"
    init_y = genjax.uniform(1., SCENE_SIZE.astype(jnp.float32)) @ "y"

    vx = genjax.normal(0., 1.) @ "vx"
    vy = genjax.normal(0., 1.) @ "vy"

    blink_rate = genjax.normal(0.1, 0.01) @ "blink_rate"
    blinking = jax.lax.select(True, False, False)
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

@genjax.scan(n=100)
@gen 
def step_firefly(firefly, carry):
    x = firefly["x"]
    y = firefly["y"]
    vx = firefly["vx"]
    vy = firefly["vy"]
    blink_rate = firefly["blink_rate"]
    blinking = firefly["blinking"]
    state_duration = firefly["state_duration"]

    vx = genjax.truncated_normal(vx, .25, -2., 2.) @ "vx"
    vy = genjax.truncated_normal(vy, .25, -2., 2.) @ "vy"

    # Switch on collision
    vx = jnp.where((x + vx > SCENE_SIZE - 1.) | (x + vx < 1.), -vx, vx)
    vy = jnp.where((y + vy > SCENE_SIZE - 1.) | (y + vy < 1.), -vy, vy)
    
    new_x = jnp.clip(x + vx, 0., SCENE_SIZE)
    new_y = jnp.clip(y + vy, 0., SCENE_SIZE)

    new_x = genjax.truncated_normal(new_x, 0.1, 0., SCENE_SIZE.astype(jnp.float32)) @ "x" 
    new_y = genjax.truncated_normal(new_y, 0.1, 0., SCENE_SIZE.astype(jnp.float32)) @ "y"

    # Update blinking - currently a finite state machine with weighted on/off
    current_blink_rate = jnp.where(blinking, 1 - blink_rate, blink_rate)
    blink = genjax.flip(current_blink_rate) @ "blink"
    
    # Keep count of duration of current state
    state_change = jnp.where(blink == blinking, 0, 1)
    new_state_duration = jnp.where(state_change, 1, state_duration + 1)

    firefly = {
        "x": new_x,
        "y": new_y,
        "vx": vx,
        "vy": vy,
        "blink_rate": blink_rate,
        "blinking": blink,
        "state_duration": new_state_duration
    }
    # Record everything
    carry.append(firefly)
    
    return firefly, firefly

@gen
def single_firefly_model():
    firefly = init_firefly() @ "init"
    firefly, chain = step_firefly(firefly, []) @ "dynamics"
    return (firefly, chain)

@jax.jit
def calculate_glow(coordinate, glow_size):
    distances = jnp.sqrt((SCENE_X - jnp.round(coordinate[1]))**2 + (SCENE_Y - jnp.round(coordinate[0]))**2)
    glow_values = jnp.exp(-(distances**2) / (2 * glow_size**2))
    return glow_values

@jax.jit
def render_frame(xs, ys, blinks, state_durations, max_glow_size):
    # Extract x, y coordinates and blinking values for the current frame
    coordinates = jnp.stack([xs, ys], axis=-1)
    blink_durations = state_durations * blinks
    glow_sizes = jnp.where(blink_durations < max_glow_size, blink_durations, max_glow_size)
    v_calculate_glow = jax.vmap(calculate_glow, in_axes=(0, 0))

    glow_values = v_calculate_glow(coordinates, glow_sizes)
    glow_grid = jnp.zeros((SCENE_SIZE, SCENE_SIZE)) # TODO: Replace with scene size

    def accumulate_glow(carry, values):
        pixelstack = jnp.array([carry, values])
        glow_grid = jnp.where(values > 0.01, jnp.max(pixelstack, axis=0), carry)
        return glow_grid, None

    # Scan across fireflies to create a composite image with additive glow
    final_glow_grid = jax.lax.scan(accumulate_glow, glow_grid, glow_values)[0]
    
    return final_glow_grid

@gen
def observe_fireflies(xs, ys, blinks, state_durations, max_glow_size=2.):
    # Vectorize the render_frame function over all timesteps
    v_render_frame = jax.vmap(render_frame, in_axes=(1, 1, 1, 1, None))
    rendered = v_render_frame(xs, ys, blinks, state_durations, max_glow_size)
    noisy_obs = genjax.normal(rendered, 0.001) @ "pixels"
    return noisy_obs

def masked_values(values, mask, fill_value=0.):
    mask = jnp.expand_dims(mask, axis=-1)
    return jnp.where(mask, values, fill_value)

@gen 
def multifirefly_model(max_fireflies, max_glow_size): 
    n_fireflies = labcat(unicat(max_fireflies), max_fireflies) @ "n_fireflies"
    masks = max_fireflies <= n_fireflies
    vmask_model = single_firefly_model.mask().vmap(in_axes=(0))
    fireflies = vmask_model(masks) @ "fireflies"

    chain = fireflies.value[1]
    xs = masked_values(chain["x"], masks)
    ys = masked_values(chain["y"], masks)
    blinks = masked_values(chain["blinking"], masks)
    state_durations = masked_values(chain["state_duration"], masks)

    observations = observe_fireflies(xs, ys, blinks, state_durations, max_glow_size) @ "observations"
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
    key = jax.random.PRNGKey(132)
    key, subkey = jax.random.split(key)
    max_fireflies = jnp.arange(1, 5)
    max_glow_size = 1.4
    steps = jnp.arange(10)

    multi_model_jit = jax.jit(multifirefly_model.importance)
    chm = C["n_fireflies"].set(3)

    tr, _ = multi_model_jit(subkey, chm, (max_fireflies, max_glow_size,))

    key, subkey = jax.random.split(key)
    tr, weight = multi_model_jit(subkey, chm, (max_fireflies, max_glow_size,))
    print(weight)
    chm = tr.get_sample()
    print(chm)
    print("Generating animation")
    frames = get_frames(chm)
    ani = animate(frames, 20)
    print("Saving animation...")
    ani.save("animations/genjax_model/maxglowsize_1.gif", writer="imagemagick", fps=20)
    plt.show()

if __name__ == "__main__":
    main()