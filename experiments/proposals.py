import genjax
import jax
import jax.numpy as jnp
from genjax import gen
from genjax import ChoiceMapBuilder as CMB

SCENE_SIZE = 32. 

@gen 
def brownian_bridge(step, total_steps, start, stop):
    T = 1
    t = (step * T) / (total_steps - 1)
    std = jnp.sqrt(t * (T - t) / T)
    position = genjax.normal(start + (t/T) * (stop - start), std)
    return position


"""
Input to the proposal is a vector of past observations and states
"""

@gen
def random_walk_step(prev_state):
    x, y, vx, vy = prev_state
    # Switch direction on collision
    vx = jnp.where((x + vx > SCENE_SIZE) | (x + vx < 0.), -vx, vx)
    vy = jnp.where((y + vy > SCENE_SIZE) | (y + vy < 0.), -vy, vy)
    new_vx = genjax.truncated_normal(vx, 0.3, 0., SCENE_SIZE) @ "vx"
    new_vy = genjax.truncated_normal(vy, 0.3, 0., SCENE_SIZE) @ "vy"
    new_x = genjax.truncated_normal(x + new_vx, 0.05, 0., SCENE_SIZE) @ "x"
    new_y = genjax.truncated_normal(y + new_vy, 0.05, 0., SCENE_SIZE) @ "y"
    blinking = genjax.bernoulli(0.01) @ "blinking"
    return (new_x, new_y, new_vx, new_vy)


@gen
def target_based_step(prev_state, obs_x, obs_y):
    prev_x, prev_y, prev_vx, prev_vy = prev_state

    valid_obs = jnp.where(obs_x > -1.)
    obs_x = obs_x[valid_obs]
    obs_y = obs_y[valid_obs]
    dists = jnp.sqrt((prev_x - obs_x)**2 + (prev_y - obs_y)**2)

    # get distribution of distances
    dist_probs = dists / jnp.sum(dists)
    closest_obs = jnp.argmin(dists)
    closest_x = obs_x[closest_obs]
    closest_y = obs_y[closest_obs]

    # sample target location
    target_idx = genjax.categorical(dist_probs)
    target_x = obs_x[target_idx]
    target_y = obs_y[target_idx]

    vx = genjax.normal(target_x - prev_y, 0.01) @ "vx"
    vy = genjax.normal(target_y - prev_y, 0.01) @ "vy"

    x = genjax.truncated_normal(target_x, 0.01, 0., SCENE_SIZE) @ "x"
    y = genjax.truncated_normal(target_y, 0.01, 0., SCENE_SIZE) @ "y"

    blinking = genjax.bernoulli(.99) @ "blinking"

    return (x, y, vx, vy)


@gen
def step_proposal(prev_state, obs_xs, obs_ys):
    """
    1. If there are no observations, update
       states by sampling a grid of velocities

    2. If there are observations:  
        2a. Sample an assignment based on proximity to 
             either the current state or the last observation
            (if it exists)
        2b. If sampling based on last observation, update
            all the observations accordingly
    """

    prev_x, prev_y, prev_vx, prev_vy = prev_state
    
    # Check if there are observations
    observed_locs = jnp.sum(obs_xs > -1.) > 0
    dynamics = genjax.switch(target_based_step, random_walk_step)
    dynamics_args = (prev_state, obs_xs[-1], obs_ys[-1])
    new_state = dynamics(observed_locs, dynamics_args, dynamics_args) @ "dynamics"

    return new_state

@gen
def init_proposal(obs_x, obs_y):
    valid_idxs = jnp.where(obs_x > -1.)
    
    xs = obs_x[valid_idxs]
    ys = obs_y[valid_idxs]

    x = genjax.uniform(0, SCENE_SIZE) @ "x"
    y = genjax.uniform(0, SCENE_SIZE) @ "y"
    vx = genjax.uniform(-2., 2.) @ "vx"
    vy = genjax.uniform(-2., 2.) @ "vy"

    return (x, y, vx, vy)

# Particle Filter:
# 1. generate trace
# 2. sample init
# 3. Score under model, score under proposal compute weight as p - q
# 4. run chain:
# 5.   propose, score, reweight, maybe resample


class JaxKey():
    def __init__(self, seed):
        self.key = jax.random.PRNGKey(seed)

    def __call__(self, n_keys=1):
        if n_keys == 1:
            _, subkey = jax.random.split(self.key)
            self.key = subkey
            return self.key
        else:
            keys = jax.random.split(self.key, n_keys)
            self.key = keys[-1]
            return keys



def test_filter():
    keygen = JaxKey(123)
    
    
