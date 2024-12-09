import jax
import jax.numpy as jnp
import genjax
from genjax import ChoiceMapBuilder as C
import matplotlib.pyplot as plt

import sys
sys.path.append('../src/')
import maskcombinator_model as mcm
from utils import *

"""
###################################
# single_firefly_chm.py
###################################

Defines several scenes in the form of choicemaps for the existing MaskCombinatorModel.
The scenes should provide examples of interesting probabilistic inferences that can be made with the model.


[Case 2] 1 vs. 2 fireflies (velocity)
----------------------------- 
- Observe two blinks at different times relatively far apart
- Whether it's a single firefly or two fireflies depends on the timing between the blinks and the velocity of the fireflies
"""

@genjax.gen
def generate_trajectory(prev_state):
    """
    Generates a noisy trajectory over T steps

    Args:
        prev_state: Tuple with the following contents:
            x (float): initial x-coordinate
            y (float): initial y-coordinate
            vx (float): x-velocity
            vy (float): y-velocity
            scene_size (int): Size of the scene
            T (int): Number of steps
    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: (x, y) trajectory
    """

    x, y, vx, vy, scene_size = prev_state
    vx = jnp.where((x + vx > scene_size) | (x + vx < 0.), -vx, vx)
    vy = jnp.where((y + vy > scene_size) | (y + vy < 0.), -vy, vy)
    vx = genjax.normal(vx, 0.1) @ "vx"
    vy = genjax.normal(vy, 0.1) @ "vy"
    x = genjax.normal(x + vx, 0.5) @ "x"
    y = genjax.normal(y + vy, 0.5) @ "y"
    return (x, y, vx, vy, scene_size)

def get_trajectory(steps: int, scene_size: float, vx=1., vy=1., key=jax.random.PRNGKey(43)):
    """
    Runs trajectory model for T steps and returns the x, y coordinates
    Args:
        steps (int): Number of steps
        scene_size (float): Size of the scene
        vx (float): x-velocity
        vy (float): y-velocity
        key (jnp.PRNGKey): random key
    """
    keys = jax.random.split(key, steps)
    init_x = jax.random.uniform(key) * scene_size
    init_y = jax.random.uniform(key) * scene_size
    trajectory = generate_trajectory.iterate(n=steps - 1).simulate(key, ((init_x, init_y, vx, vy, scene_size,),))
    retvals = trajectory.get_retval()
    xs, ys = retvals[0], retvals[1]
    return xs, ys

def n_similar_trajectories(N: int, T: int, scene_size: float, vx=1., vy=1., noise_std=0.1, key=jax.random.PRNGKey(43)):
    """
    Generates a trajectory over T steps and adds noise to generate N - 1 similar trajectories

    Args:
        N (int): Number of trajectories
        T (int): Number of steps
        scene_size (float): Size of the scene
        vx (float): x-velocity
        vy (float): y-velocity
        noise_std (float): standard deviation of noise
        key (jnp.PRNGKey): random key
    Returns
        Tuple[jnp.ndarray, jnp.ndarray]: (x, y) trajectories
    """
    xs, ys = get_trajectory(T, scene_size, vx, vy, key)
    xs = xs.squeeze()
    ys = ys.squeeze()
    trajectories = [(xs, ys)]
    for i in range(N-1):
        # duplicate trajectory and add some noise
        key = jax.random.split(key)[1]
        new_xs = jnp.clip(xs + jax.random.normal(key, shape=(T,)) * noise_std, 0, scene_size)
        new_ys = jnp.clip(ys + jax.random.normal(key, shape=(T,)) * noise_std, 0, scene_size)

        trajectories.append((new_xs, new_ys))
    return trajectories


def generate_assignments(observed_xs, observed_ys):
    """
    Generates a set of assignments for the observed x, y coordinates

    Args:
        observed_xs (jnp.ndarray): observed x-coordinates shaped (T, N)
        observed_ys (jnp.ndarray): observed y-coordinates shaped (T, N)
    Returns:
        jnp.ndarray: assignments
    """
    n = observed_xs.shape[0]
    assignments = jnp.zeros((n, n)).astype(jnp.bool)
    for i in range(n):
        for j in range(n):
            assignments = jax.ops.index_update(assignments, (i, j), (observed_xs[i] == observed_xs[j]) & (observed_ys[i] == observed_ys[j]))
    return assignments


def one_vs_two_blink_rate_dependent_inference(blink_interval, steps, scene_size=32., max_fireflies=2):
    """    
    [Case 1] 1 vs. 2 fireflies (base_blink_rate)
    -----------------------------
    - Observe high frequency of blinks along a single trajectory
    - Posterior likelihood is a function over prior on blink rate
        - the lower the base blink rate, the more likely it's two different fireflies

    
    """
    # define the scene
    xs, ys = get_trajectory(steps, scene_size)

    n_blinks = steps // blink_interval
    blinks = jnp.zeros((steps, 2)).astype(jnp.bool)
    blinks = blinks.at[jnp.arange(0, steps, blink_interval), 0].set(True)
    observed_xs = jnp.where(blinks, jnp.vstack([xs, xs]).T, -1.)
    observed_ys = jnp.where(blinks, jnp.vstack([ys, ys]).T, -1.)

    # Generate assignments where it's a single firefly
    single_firefly_chm = C["n_fireflies"].set(1)
    single_firefly_chm = single_firefly_chm | C["steps", :, "dynamics", :, "blinking"].set(blinks)
    # single_firefly_chm = single_firefly_chm | C["steps", :, "dynamics", :, "x"].set(observed_xs)
    # single_firefly_chm = single_firefly_chm | C["steps", :, "dynamics", :, "y"].set(observed_ys)
    single_firefly_chm = single_firefly_chm | C["steps", :, "observations", "observed_xs", :].set(observed_xs)
    single_firefly_chm = single_firefly_chm | C["steps", :, "observations", "observed_ys", :].set(observed_ys)
    
    # Generate assignments where it's two fireflies
    two_fireflies_chm = C["n_fireflies"].set(2)
    
    # Alternate the blinks between the two fireflies
    blinks = jnp.zeros((steps, 2)).astype(jnp.bool)
    blinks = blinks.at[jnp.arange(0, steps, blink_interval * 2), 0].set(True)
    blinks = blinks.at[jnp.arange(blink_interval // 2, steps, blink_interval * 2), 1].set(True)

    observed_xs = jnp.where(blinks, jnp.vstack([xs, xs]).T, -1.)
    observed_ys = jnp.where(blinks, jnp.vstack([ys, ys]).T, -1.)

    two_fireflies_chm = two_fireflies_chm | C["steps", :, "dynamics", :, "blinking"].set(blinks)
    # two_fireflies_chm = two_fireflies_chm | C["steps", :, "dynamics", :, "x"].set(jnp.stack([observed_xs1, observed_xs2]))
    # two_fireflies_chm = two_fireflies_chm | C["steps", :, "dynamics", :, "y"].set(jnp.stack([observed_ys1, observed_ys2]))
    two_fireflies_chm = two_fireflies_chm | C["steps", :, "observations", "observed_xs", :].set(observed_xs)
    two_fireflies_chm = two_fireflies_chm | C["steps", :, "observations", "observed_ys", :].set(observed_ys)

    return single_firefly_chm, two_fireflies_chm, (xs, ys)


if __name__=="__main__":
    key = jax.random.PRNGKey(4341)
    TIME_STEPS = 30
    scene_size = 32.
    blink_interval = 2
    single_firefly_chm, two_fireflies_chm, (xs, ys) = one_vs_two_blink_rate_dependent_inference(
                                            blink_interval, TIME_STEPS, scene_size=scene_size)  
    importance_jit = jax.jit(mcm.multifirefly_model.importance)
    key, subkey = jax.random.split(key)
    time_mask = jnp.arange(TIME_STEPS) < TIME_STEPS
    max_fireflies = jnp.arange(1, 3)
    single_tr, single_w = importance_jit(subkey, single_firefly_chm, (max_fireflies, time_mask,))
    two_tr, two_w = importance_jit(subkey, two_fireflies_chm, (max_fireflies, time_mask,))

    print(single_w, two_w)
    observed_xs, observed_ys = get_observations(single_tr.get_sample())

    anim = scatter_animation(observed_xs, observed_ys)
    # Add barplot with relative scores
    fig, ax = plt.subplots()
    ax.bar(["Single firefly", "Two fireflies"], [single_w, two_w])
    plt.show()