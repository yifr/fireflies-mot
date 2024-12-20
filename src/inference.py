import genjax
import jax
import jax.numpy as jnp
from genjax import gen, flip, normal, uniform
from genjax import truncated_normal as truncnorm
from genjax import ChoiceMapBuilder as CMB

import sys
sys.path.append("../src/")
from distributions import *

SCENE_SIZE = 32. 
MAX_VELOCITY = 5.
MIN_VELOCITY = -5.


##################################
# INITIALIZATION PROPOSALS
##################################
@gen
def init_firefly_at_random():
    init_x = uniform(1., SCENE_SIZE) @ "x"
    init_y = uniform(1., SCENE_SIZE) @ "y"

    vx = truncnorm(0., .5, MIN_VELOCITY, MAX_VELOCITY) @ "vx"
    vy = truncnorm(0., .5, MIN_VELOCITY, MAX_VELOCITY) @ "vy"

    blink_rate = normal(0.1, 0.01) @ "blink_rate"
    blinking = jnp.bool(0)
    #state_duration = jax.lax.select(True, 0, 0)

    firefly = {
        "x": init_x,
        "y": init_y,
        "vx": vx,
        "vy": vy,
        "blink_rate": blink_rate,
        "blinking": blinking,
    }

    return firefly

@gen
def init_firefly_at_loc(obs_x, obs_y):
    """
    Takes in observed locations and initializes 
    fireflies if the locations are in bounds
    """
    vx = truncnorm(0., .3, MIN_VELOCITY, MAX_VELOCITY) @ "vx"
    vy = truncnorm(0., .3, MIN_VELOCITY, MAX_VELOCITY) @ "vy"

    # If obs_x / obs_y are < 0, sample starting position uniformly
    is_valid_x = obs_x > 0.
    is_valid_y = obs_y > 0.
    x = truncnorm.or_else(uniform)(is_valid_x, (obs_x - vx, 0.01, 0., SCENE_SIZE), (0., SCENE_SIZE)) @ "x"
    y = truncnorm.or_else(uniform)(is_valid_y, (obs_y - vy, 0.01, 0., SCENE_SIZE), (0., SCENE_SIZE)) @ "y"

    blink_rate = normal(0.1, 0.01) @ "blink_rate"
    blinking = jnp.bool(0)
    
    firefly = {
        "x": x,
        "y": y,
        "vx": vx,
        "vy": vy,
        "blink_rate": blink_rate,
        "blinking": blinking,
    }

    return firefly

@gen 
def model_init_fireflies(max_fireflies):
    n_fireflies = labcat(unicat(max_fireflies), max_fireflies) @ "n_fireflies"
    masks = jnp.array(max_fireflies <= n_fireflies)
    init_states = init_firefly_at_random.mask().vmap(in_axes=(0))(masks) @ "init"
    return init_states


@gen 
def proposal_init_fireflies(max_fireflies, x_obs, y_obs):
    """
    max_fireflies: jnp.array of the form [1, 2, 3, ..., max_fireflies]
    x_obs, y_obs: (max_fireflies,) array with observations. Valid observations 
          are anything inside the scene limits
    """
    num_valid_obs = jnp.sum(jnp.where(x_obs > -1., 1, 0))
    firefly_probs = jnp.where(max_fireflies < num_valid_obs + 1, 0.01, 1.) 
    firefly_probs = firefly_probs / jnp.sum(firefly_probs)
    n_fireflies = labcat(unicat(firefly_probs), max_fireflies) @ "n_fireflies"
    masks = jnp.array(max_fireflies <= n_fireflies)
    init_states = init_firefly_at_loc.mask().vmap(in_axes=(0, 0, 0))(masks, x_obs, y_obs) @ "init"
    return init_states


##################################
# DYNAMICS 
#################################

@gen 
def prior_dynamics_step(prev_state):
    """
    Single step dynamics for an individual prev_state.
    Random walk with small drift on velocity and position
    truncated to min/max velocity and position in scene bounds

    Args:
        prev_state: dictionary of prev_state 
    Returns: 
        prev_state: dictionary of updated state
    """
    prev_x = prev_state["x"]
    prev_y = prev_state["y"]
    prev_vx = prev_state["vx"]
    prev_vy = prev_state["vy"]
    blink_rate = prev_state["blink_rate"]

    # Sample a new trajectory
    new_vx = genjax.normal(prev_vx, .3) @ "vx"
    new_vy = genjax.normal(prev_vy, .3) @ "vy"

    # Switch direction on collision
    new_vx = jnp.where((prev_x + new_vx >= SCENE_SIZE - 1.) | (prev_x + new_vx <= 1.), -new_vx, new_vx)
    new_vy = jnp.where((prev_y + new_vy >= SCENE_SIZE - 1.) | (prev_y + new_vy <= 1.), -new_vy, new_vy)

    # Clip new position inside scene
    new_x = jnp.clip(prev_x + new_vx, 0., SCENE_SIZE)
    new_y = jnp.clip(prev_y + new_vy, 0., SCENE_SIZE)

    # Add some noise
    new_x = truncnorm(new_x, 0.01, 0., SCENE_SIZE) @ "x" 
    new_y = truncnorm(new_y, 0.01, 0., SCENE_SIZE) @ "y"
    
    # Update blinking 
    blinking = flip(blink_rate) @ "blinking"

    new_state = {
        "x": new_x,
        "y": new_y,
        "vx": new_vx,
        "vy": new_vy,
        "blink_rate": blink_rate,
        "blinking": blinking,
    }
    
    return new_state

def calculate_distances_from_pos(position, observations):
    """
    Args:
        position: (2,) array of (x, y) position
        observations: (2, n_objects) matrix of (x, y) observations
                        where invalid observations are < 0.
    Returns:
        distances: (n_objects, n_objects) matrix with L2 distances for each object, observation pair
                    where distances to invalid observations is -jnp.inf
    """    
    # Expand dims for broadcasting and compute norm
    pos_expanded = jnp.expand_dims(position, 1)  # (2, 1)
    diff = pos_expanded - observations
    distances = jnp.linalg.norm(diff, axis=0)
    valid = jnp.all(observations > 0., axis=0)
    return jnp.where(valid , distances, jnp.inf)


@gen
def proposal_dynamics_step(prev_state, obs_x, obs_y):
    prev_x = prev_state["x"]
    prev_y = prev_state["y"]
    prev_vx = prev_state["vx"]
    prev_vy = prev_state["vy"]
    blink_rate = prev_state["blink_rate"]
    blinking = prev_state["blinking"]

    observed_positions = jnp.stack([obs_x, obs_y], axis=0)
    distances = calculate_distances_from_pos(jnp.stack([prev_x, prev_y]), observed_positions)
    nearby_blinks = jnp.any(distances < MAX_VELOCITY)
    blinking = flip.or_else(flip)(nearby_blinks, (1.,), (0.0,)) @ "blinking"
    
    index = jax.lax.cond(nearby_blinks, lambda: jnp.argmin(distances), lambda: -1) # index of closest blink
    
    target_vx = obs_x[index] - prev_x
    target_vy = obs_y[index] - prev_y
    vx_diff = jnp.abs(target_vx - prev_vx)
    vy_diff = jnp.abs(target_vy - prev_vy)
    vx = normal.or_else(normal)(nearby_blinks, (target_vx, 0.1), 
                                                    (prev_vx, .3)) @ "vx"
    vy = normal.or_else(normal)(nearby_blinks, (target_vy, 0.1), 
                                                    (prev_vy, .3)) @ "vy"
    
    # Switch direction on collision
    vx = jnp.where((prev_x + vx >= SCENE_SIZE - 1.) | (prev_x + vy <= 1.), -vx, vx)
    vy = jnp.where((prev_y + vy >= SCENE_SIZE - 1.) | (prev_y + vy <= 1.), -vy, vy)

    x = jnp.clip(prev_x + vx, 0., SCENE_SIZE)
    y = jnp.clip(prev_y + vy, 0., SCENE_SIZE)

    x = truncnorm(x, 0.01, 0., SCENE_SIZE) @ "x"
    y = truncnorm(y, 0.01, 0., SCENE_SIZE) @ "y"
    
    new_state = {
        "x": x,
        "y": y,
        "vx": vx,
        "vy": vy,
        "blinking": blinking,
        "blink_rate": blink_rate
    }
    return new_state



def assign_observations(positions, observations, max_velocity=5., inf_val=jnp.inf):
    """
    Greedily assigns each observation to closest position, then fills remaining positions.
    Args:
        positions: (2, n_pos) array of (x,y) positions
        observations: (2, n_obs) array of (x,y) observations, invalid < 0
    Returns:
        indices: (n_pos,) array mapping each position to observation index
    """
    # Calculate pairwise distances
    pos_expanded = jnp.expand_dims(positions, 2)  
    obs_expanded = jnp.expand_dims(observations, 1)
    dists = jnp.linalg.norm(pos_expanded - obs_expanded, axis=0)
    
    # Mask invalid observations
    valid_obs = jnp.all(observations > 0., axis=0)
    dists = jnp.where(valid_obs, dists, inf_val)
    dists = jnp.where(dists <= max_velocity, dists, inf_val)

    n_pos = positions.shape[1]
    assignments = jnp.zeros((n_pos,), dtype=jnp.int32) - 1  # -1 means unassigned
    used_obs = jnp.zeros((observations.shape[1],), dtype=bool)
    
    def step(carry, _):
        dists, assignments, used_obs = carry
        masked_dists = jnp.where(used_obs, inf_val, dists)
        unassigned = assignments == -1
        masked_dists = jnp.where(jnp.expand_dims(unassigned, 1), masked_dists, inf_val)
        
        idx = jnp.argmin(masked_dists.flatten())
        pos_idx, obs_idx = jnp.unravel_index(idx, dists.shape)
        
        # Update assignments and mark observation as used
        new_assignments = assignments.at[pos_idx].set(obs_idx)
        new_used = used_obs.at[obs_idx].set(True)
        
        return (dists, new_assignments, new_used), None
    
    (_, final_assignments, _), _ = jax.lax.scan(
        step, 
        (dists, assignments, used_obs),
        xs=None,
        length=n_pos
    )
    
    return final_assignments



@gen
def greedy_proposal_dynamics_step(prev_state, obs_x, obs_y, assignment):
    prev_x = prev_state["x"]
    prev_y = prev_state["y"]
    prev_vx = prev_state["vx"]
    prev_vy = prev_state["vy"]
    blink_rate = prev_state["blink_rate"]
    blinking = prev_state["blinking"]
    
    nearby_blinks = jax.lax.cond(assignment > -1, lambda: True, lambda: False)
    blinking = flip.or_else(flip)(nearby_blinks, (1.,), (0.,)) @ "blinking"
    
    target_vx = obs_x[assignment] - prev_x
    target_vy = obs_y[assignment] - prev_y
    
    vx = normal.or_else(normal)(nearby_blinks, (target_vx, 0.05), 
                                                    (prev_vx, .3)) @ "vx"
    vy = normal.or_else(normal)(nearby_blinks, (target_vy, 0.05), 
                                                    (prev_vy, .3)) @ "vy"
    
    # Switch direction on collision
    vx = jnp.where((prev_x + vx >= SCENE_SIZE - 1.) | (prev_x + vy <= 1.), -vx, vx)
    vy = jnp.where((prev_y + vy >= SCENE_SIZE - 1.) | (prev_y + vy <= 1.), -vy, vy)

    x = jnp.clip(prev_x + vx, 0., SCENE_SIZE)
    y = jnp.clip(prev_y + vy, 0., SCENE_SIZE)

    x = truncnorm(x, 0.01, 0., SCENE_SIZE) @ "x"
    y = truncnorm(y, 0.01, 0., SCENE_SIZE) @ "y"
    
    new_state = {
        "x": x,
        "y": y,
        "vx": vx,
        "vy": vy,
        "blinking": blinking,
        "blink_rate": blink_rate
    }
    return new_state


@gen
def mutually_exclusive_proposal_dynamics(states, obs_x, obs_y):
    """
    States is a masked object with a dict of state values inside
    masks is an (n_fireflies,) array of mask vals
    obs_x and obs_y are (n_fireflies,) vectors of observations
    """
    masks = states.flag
    positions = jnp.array([states.value["x"], states.value["y"]])
    observations = jnp.array([obs_x, obs_y])
    assignments = assign_observations(positions, observations)
    proposal_fn = greedy_proposal_dynamics_step.mask().vmap(in_axes=(0, 0, None, None, 0))
    new_states = proposal_fn(masks, states.value, obs_x, obs_y, assignments) @ "steps"
    return new_states


@gen
def masked_proposal_dynamics(states, obs_x, obs_y):
    """
    States is a masked object with a dict of state values inside
    masks is an (n_fireflies,) array of mask vals
    obs_x and obs_y are (n_fireflies,) vectors of observations
    """
    masks = states.flag
    proposal_fn = proposal_dynamics_step.mask().vmap(in_axes=(0, 0, None, None))
    new_states = proposal_fn(masks, states.value, obs_x, obs_y) @ "steps"
    return new_states


@gen
def masked_model_dynamics(states):
    """
    States is an (n_fireflies,) array of dicts
    masks is an (n_fireflies,) array of mask vals
    obs_x and obs_y are (n_fireflies,) vectors of observations
    """
    masks = states.flag
    model_fn = prior_dynamics_step.mask().vmap(in_axes=(0, 0))
    new_states = model_fn(masks, states.value) @ "steps"
    return new_states

@gen 
def observation_likelihood(xs, ys, blinks):
    observed_xs = jnp.full_like(xs, -10.)
    observed_ys = jnp.full_like(ys, -10.)
    observed_xs = jnp.where(blinks, xs, observed_xs)
    observed_ys = jnp.where(blinks, ys, observed_ys)
    
    observed_xs = genjax.normal(observed_xs, .25) @ "observed_xs"
    observed_ys = genjax.normal(observed_ys, .25) @ "observed_ys"

    return jnp.stack([observed_xs, observed_ys])