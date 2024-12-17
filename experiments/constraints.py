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
    vx = genjax.normal(vx, 0.3) @ "vx"
    vy = genjax.normal(vy, 0.3) @ "vy"
    x = genjax.truncated_normal(x + vx, 0.01, 0., jnp.float32(scene_size)) @ "x"
    y = genjax.truncated_normal(y + vy, 0.01, 0., jnp.float32(scene_size)) @ "y"
    return (x, y, vx, vy, scene_size)

def get_trajectory(steps: int, scene_size: float, init_x=None, init_y=None, vx=1., vy=1., key=jax.random.PRNGKey(43)):
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
    if init_x is None:
        init_x = jax.random.uniform(key) * scene_size
    if init_y is None:
        init_y = jax.random.uniform(key) * scene_size
    trajectory = generate_trajectory.iterate(n=int(steps)).simulate(key, ((init_x, init_y, vx, vy, scene_size,),))
    score = trajectory.get_score()
    retvals = trajectory.get_retval()
    xs, ys = retvals[0][:-1], retvals[1][:-1]
    return (xs, ys), score

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
    (xs, ys), score = get_trajectory(T, scene_size, vx, vy, key)
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


def trajectory_between(start_point, end_point, num_steps, noise_scale=1.0, seed=None):
    """
    Generate a random walk trajectory between two points that reaches the destination in exactly T steps.
    
    Parameters:
    -----------
    start_point : tuple of (float, float)
        Starting (x, y) coordinates
    end_point : tuple of (float, float)
        Ending (x, y) coordinates
    num_steps : int
        Number of steps in the trajectory
    noise_scale : float
        Scale factor for the random noise (default=1.0)
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    xs : numpy array
        x-coordinates of the trajectory
    ys : numpy array
        y-coordinates of the trajectory
    """
    if seed is not None:
        np.random.seed(seed)
    
    x0, y0 = start_point
    x1, y1 = end_point
    
    # Initialize arrays for storing coordinates
    xs = np.zeros(num_steps + 1)
    ys = np.zeros(num_steps + 1)

    # Set initial points separately
    xs[0] = x0
    ys[0] = y0
    
    noise_x = np.random.normal(0, noise_scale, num_steps)
    noise_y = np.random.normal(0, noise_scale, num_steps)
    
    for t in range(1, num_steps + 1):
        progress = t / num_steps
        base_x = x0 + (x1 - x0) * progress
        base_y = y0 + (y1 - y0) * progress
        
        if t < num_steps:
            noise_factor = 1 - progress
            current_noise_x = noise_x[t-1] * noise_factor
            current_noise_y = noise_y[t-1] * noise_factor
            xs[t] = base_x + current_noise_x
            ys[t] = base_y + current_noise_y
        else:
            xs[t] = x1
            ys[t] = y1
    
    return xs[:-1], ys[:-1]

def constrained_trajectory(observed_xs, observed_ys, scene_size=32., noise_scale=1.0, key=jax.random.PRNGKey(43)):
    """
    Fills in reasonable values for any unobserved x, y coordinates
    Args:
        observed_xs: (steps,) sized array of observed x-coordinates for a single firefly
        observed_ys: (steps,) sized array of observed y-coordinates for a single firefly
        scene_size: size of the scene
        noise_scale: scale of random noise in the interpolation
    Returns:
        xs: (steps,) sized array of x-coordinates filled in
        ys: (steps,) sized array of y-coordinates filled in
    """
    # Convert to numpy for easier handling
    observed_xs = np.array(observed_xs)
    observed_ys = np.array(observed_ys)
    
    xs = observed_xs.copy()
    ys = observed_ys.copy()

    # Find indices where we have observations (x > -1)
    observation_indices = np.where(observed_xs > 0)[0]
    if len(observation_indices) == 0:
        # No observations, generate random walk from start to end
        (xs, ys) = get_trajectory(len(observed_xs), 32., key=key) 
        return xs, ys

    # Handle case where first observation isn't at t=0
    if observation_indices[0] > 0:
        first_x = observed_xs[observation_indices[0]]
        first_y = observed_ys[observation_indices[0]]
        steps = observation_indices[0]
        # Set start point based on velocity and steps to first observation
        # Make sure it's inside the scene    
        walk_xs, walk_ys = get_trajectory(steps, scene_size, first_x, first_y, key=key)
        xs[:steps] = walk_xs[::-1]
        ys[:steps] = walk_ys[::-1]

    # Fill in trajectories between each pair of observations
    for i in range(len(observation_indices)-1):
        start_idx = observation_indices[i]
        end_idx = observation_indices[i+1]
        steps = end_idx - start_idx
        
        if steps <= 1:
            continue
            
        start_point = (observed_xs[start_idx], observed_ys[start_idx])
        end_point = (observed_xs[end_idx], observed_ys[end_idx])
        
        walk_xs, walk_ys = trajectory_between(start_point, end_point, steps, noise_scale)
        xs[start_idx:end_idx] = jnp.clip(walk_xs, 0., scene_size)
        ys[start_idx:end_idx] = jnp.clip(walk_ys, 0., scene_size)
    
    # Handle case where last observation isn't at the end
    if observation_indices[-1] < len(observed_xs) - 1:
        last_x = observed_xs[observation_indices[-1]]
        last_y = observed_ys[observation_indices[-1]]
        # Generate random walk from last observation to scene edge
        steps = len(observed_xs) - observation_indices[-1] - 1
        if steps > 0:
            (walk_xs, walk_ys) = get_trajectory(steps, scene_size, last_x, last_y, key=key)
            xs[observation_indices[-1]+1:] = jnp.clip(walk_xs, 0., scene_size)
            ys[observation_indices[-1]+1:] = jnp.clip(walk_ys, 0., scene_size)
    
    return xs, ys


def generate_full_trajectory_constraints(observed_xs, observed_ys, scene_size=32., noise_scale=1.0, key=jax.random.PRNGKey(43)):
    """
    observed_xs: (steps, N) sized array of observed x-coordinates for N fireflies
    observed_ys: (steps, N) sized array of observed y-coordinates for N fireflies

    Returns:
    xs: (steps, N) sized array of x-coordinates filled in
    ys: (steps, N) sized array of y-coordinates filled in
    """
    steps, n = observed_xs.shape
    xs = np.zeros_like(observed_xs)
    ys = np.zeros_like(observed_ys)
    vxs = np.zeros_like(observed_xs) 
    vys = np.zeros_like(observed_ys)

    for i in range(n):
        xs[:, i], ys[:, i] = constrained_trajectory(observed_xs[:, i], observed_ys[:, i], scene_size, noise_scale, key=key)
        # v_t = x_{t+1} - x_t
        vxs[:-1, i] = xs[1:, i] - xs[:-1, i]
        vys[:-1, i] = ys[1:, i] - ys[:-1, i]

    if steps > 1:
        vxs[-1] = 0.5
        vys[-1] = 0.5
    
    return xs, ys, vxs, vys


def get_trajectory_constraints(observed_xs, observed_ys, existing_constraints=None, scene_size=32., noise_scale=1.0, key=jax.random.PRNGKey(43)):
    """
    Evaluate the likelihood of the observed trajectory given the model

    Args:
    model: MaskCombinatorModel
    observed_xs: (steps, N) sized array of observed x-coordinates for N fireflies
    observed_ys: (steps, N) sized array of observed y-coordinates for N fireflies
    scene_size: size of the scene
    noise_scale: scale of random noise in the interpolation

    Returns:
    likelihood: float
    """
    xs, ys, vxs, vys = generate_full_trajectory_constraints(observed_xs, observed_ys, scene_size, noise_scale, key=key)

    if existing_constraints is None:
        chm = C.n()
    else:
        chm = existing_constraints
        
    chm = chm | C["steps", :, "dynamics", :, "x"].set(jnp.array(xs)) 
    chm = chm | C["steps", :, "dynamics", :, "y"].set(jnp.array(ys))
    chm = chm | C["steps", :, "dynamics", :, "vx"].set(jnp.array(vxs))
    chm = chm | C["steps", :, "dynamics", :, "vy"].set(jnp.array(vys))
    chm = chm | C["steps", :, "observations", "observed_xs", :].set(observed_xs)
    chm = chm | C["steps", :, "observations", "observed_ys", :].set(observed_ys)

    # Set initial conditions
    init_x = xs[0] - vxs[0]
    init_y = ys[0] - vys[0]
    init_vx = vxs[0]
    init_vy = vys[0]

    chm = chm | C["init", :, "x"].set(init_x)
    chm = chm | C["init", :, "y"].set(init_y)
    chm = chm | C["init", :, "vx"].set(init_vx)
    chm = chm | C["init", :, "vy"].set(init_vy)

    # Set blinks
    blinking = jnp.where(observed_xs > -1, True, False)
    blink_rates = blinking.sum(axis=0) / len(observed_xs)
    chm = chm | C["steps", :, "dynamics", :, "blinking"].set(blinking)
    chm = chm | C["init", :, "blink_rate"].set(blink_rates)

    return chm


def generate_assignments(observed_xs, observed_ys, key=jax.random.PRNGKey(43)):
    """
    Samples an assignments for the observed x, y coordinates

    Args:
        observed_xs (jnp.ndarray): observed x-coordinates shaped (T, N)
        observed_ys (jnp.ndarray): observed y-coordinates shaped (T, N)
    Returns:
        jnp.ndarray: assignments
    """
    t, n = observed_xs.shape
    num_observations = jnp.sum(observed_xs > 0)
    total_possible_assignments = num_observations * (num_observations - 1) // 2


def one_vs_two_blink_rate_dependent_inference(blink_interval, steps, scene_size=32., max_fireflies=2):
    """    
    [Case 1] 1 vs. 2 fireflies (base_blink_rate)
    --------------------------------------------
    - Observe high frequency of blinks along a single trajectory
    - Posterior likelihood is a function over prior on blink rate
        - the lower the base blink rate, the more likely it's two different fireflies

    
    """
    # define the scene
    trajectories = n_similar_trajectories(2, steps, scene_size, noise_std=0.1)

    xs, ys = trajectories[0]
    n_blinks = steps // blink_interval
    blinks = jnp.zeros((steps, 2)).astype(jnp.bool)
    blinks = blinks.at[jnp.arange(0, steps, blink_interval), 0].set(True)
    observed_xs = jnp.where(blinks, jnp.vstack([xs, xs]).T, -1.)
    observed_ys = jnp.where(blinks, jnp.vstack([ys, ys]).T, -1.)

    # Generate assignments where it's a single firefly
    single_firefly_chm = C["n_fireflies"].set(1)
    single_firefly_chm = single_firefly_chm | C["steps", :, "dynamics", :, "blinking"].set(blinks)
    for t, observed_x in enumerate(observed_xs):
        for n, x in enumerate(observed_x):
            if x > 0:
                single_firefly_chm = single_firefly_chm | C["steps", t, "dynamics", n, "x"].set(x)
                single_firefly_chm = single_firefly_chm | C["steps", t, "dynamics", n, "y"].set(observed_ys[t, n])

    single_firefly_chm = single_firefly_chm | C["steps", :, "observations", "observed_xs", :].set(observed_xs)
    single_firefly_chm = single_firefly_chm | C["steps", :, "observations", "observed_ys", :].set(observed_ys)
    

    # Generate assignments where it's two fireflies
    two_fireflies_chm = C["n_fireflies"].set(2)
    
    # Alternate the blinks between the two fireflies
    blinks = jnp.zeros((steps, 2)).astype(jnp.bool)
    blinks = blinks.at[jnp.arange(0, steps, blink_interval), 0].set(True)
    blinks = blinks.at[jnp.arange(blink_interval // 2, steps, blink_interval), 1].set(True)

    xs = [t[0] for t in trajectories]
    ys = [t[1] for t in trajectories]
    observed_xs = jnp.where(blinks, jnp.vstack(xs).T, -1.)
    observed_ys = jnp.where(blinks, jnp.vstack(ys).T, -1.)

    for t, observed_x in enumerate(observed_xs):
        for n, x in enumerate(observed_x):
            if x > 0:
                two_fireflies_chm = two_fireflies_chm | C["steps", t, "dynamics", n, "x"].set(x)
                two_fireflies_chm = two_fireflies_chm | C["steps", t, "dynamics", n, "y"].set(observed_ys[t, n])

    two_fireflies_chm = two_fireflies_chm | C["steps", :, "dynamics", :, "blinking"].set(blinks)
    two_fireflies_chm = two_fireflies_chm | C["steps", :, "observations", "observed_xs", :].set(observed_xs)
    two_fireflies_chm = two_fireflies_chm | C["steps", :, "observations", "observed_ys", :].set(observed_ys)

    return single_firefly_chm, two_fireflies_chm, (xs, ys)


if __name__=="__main__":
    key = jax.random.PRNGKey(1412)
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
    gt_xs, gt_ys = get_gt_locations(two_tr.get_sample())

    anim = scatter_animation(observed_xs, observed_ys, gt_xs, gt_ys)
    # Add barplot with relative scores
    fig, ax = plt.subplots()
    ax.bar(["Single firefly", "Two fireflies"], [single_w, two_w])
    plt.show()