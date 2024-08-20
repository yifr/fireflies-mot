import jax
import jax.numpy as jnp
import genjax
from genjax import gen
from genjax import ChoiceMapBuilder as C
from genjax import ExactDensity, Pytree
from tensorflow_probability.substrates import jax as tfp
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation
from distributions import *
tfd = tfp.distributions


""" Constant values """
TIME_STEPS = 100
    
""" Define Distributions """

@gen
def init_firefly(scene_size):
    init_x = genjax.uniform(1., float(scene_size.const)) @ "x"
    init_y = genjax.uniform(1., float(scene_size.const)) @ "y"
    blink_rate = genjax.normal(0.1, 0.01) @ "blink_rate"
    blinking = 0
    return (init_x, init_y, blink_rate, blinking) 

@genjax.scan(n=TIME_STEPS)
@gen 
def step_firefly(firefly, carry):
    x, y, blink_rate, blinking = firefly
    new_x = genjax.normal(x, 1.) @ "x"
    new_y = genjax.normal(y, 1.) @ "y"
    current_blink_rate = jnp.where(blinking, 1 - blink_rate, blink_rate)
    blink = genjax.flip(current_blink_rate) @ "blink"
    carry.append([new_x, new_y, blink])
    return (new_x, new_y, blink_rate, blink), carry

@gen
def single_firefly_model(scene_size):
    firefly = init_firefly(scene_size) @ "init"
    firefly, chain = step_firefly(firefly, []) @ "dynamics"
    return (firefly, chain)

@jax.jit
def calculate_glow(coordinate, glow_size, scene_size):
    y, x = jnp.mgrid[0:64, 0:64]
    distances = jnp.sqrt((x - coordinate[1])**2 + (y - coordinate[0])**2)
    # Use gaussian glow model
    glow_values = jnp.exp(-(distances**2) / (2 * glow_size**2))
    
    return glow_values

@jax.jit
def render_frame(xs, ys, blinks, scene_size, base_glow_size):
    # Extract x, y coordinates and blinking values for the current frame
    coordinates = jnp.stack([xs, ys], axis=-1)
    glow_sizes = base_glow_size * blinks
    v_calculate_glow = jax.vmap(calculate_glow, in_axes=(0, 0, None))

    glow_values = v_calculate_glow(coordinates, glow_sizes, scene_size)
    glow_grid = jnp.zeros((64, 64))

    def accumulate_glow(carry, values):
        glow_grid = carry + values
        return glow_grid, None

    final_glow_grid = jax.lax.scan(accumulate_glow, glow_grid, glow_values)[0]
    
    return final_glow_grid

@gen
def observe_fireflies(xs, ys, blinks, scene_size, base_glow_size=1.):
    # Vectorize the render_frame function over all timesteps
    v_render_frame = jax.vmap(render_frame, in_axes=(1, 1, 1, None, None))
    rendered = v_render_frame(xs, ys, blinks, scene_size, base_glow_size)
    noisy_obs = genjax.normal(rendered, 0.0001) @ "pixels"
    return noisy_obs

@gen 
def multifirefly_model(scene_size, max_fireflies):
    n_fireflies = labcat(unicat(max_fireflies), max_fireflies) @ "n_fireflies"
    masks = max_fireflies <= n_fireflies
    multifirefly_model = genjax.MaskCombinator(single_firefly_model).vmap(in_axes=(0, None))
    fireflies = multifirefly_model(masks, scene_size) @ "fireflies"

    masks = jnp.where(fireflies.flag, masks, 0)
    chain = jnp.array(fireflies.value[1]).squeeze()
    xs, ys, blinks = chain[0, :, :], chain[1, :, :], chain[2, :, :] # N, T
    xs = xs[masks, :]
    ys = ys[masks, :]
    blinks = blinks[masks, :]
    xs_shape = xs.shape

    observations = observe_fireflies(xs, ys, blinks, scene_size.const) @ "observations"
    return fireflies, observations

key = jax.random.PRNGKey(878)
key, subkey = jax.random.split(key)
fireflies = jnp.arange(1, 5)
scene_size = genjax.Pytree.const(64)
steps = jnp.arange(10)

multi_model_jit = jax.jit(multifirefly_model.simulate)
tr = multi_model_jit(subkey, (scene_size, fireflies,))

key, subkey = jax.random.split(key)
tr = multi_model_jit(subkey, (scene_size, fireflies,))
chm = tr.get_sample()

def get_frames(chm):
    observations = chm["observations", "pixels"]
    frames = []
    for i in range(observations.shape[0]):
        frames.append(observations[i])
    return frames

def animate(frames, fps):
    fig, ax = plt.subplots()
    def update(frame):
        ax.clear()
        ax.set_title(f"Frame {frame}")
        ax.imshow(frames[frame], vmin=0, vmax=1, cmap="hot")
    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000/fps)
    return ani


print("Generating animation")
frames = get_frames(chm)
ani = animate(frames, 20)
print("Saving animation...")
ani.save("firefly_simulation.gif", writer="imagemagick", fps=20)
plt.show()