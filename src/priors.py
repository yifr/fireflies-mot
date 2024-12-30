from genjax import gen, normal, flip, uniform, categorical
from genjax import truncated_normal as truncnorm
from distributions import *
import itertools
import jax
from config import SCENE_SIZE, MIN_VELOCITY, MAX_VELOCITY, BLINK_MEAN, BLINK_STD

VELOCITY_STD = .3
POSITION_STD = .01

##################################
# INITIALIZATION PROPOSALS
##################################

@gen
def init_firefly_at_random():
    init_x = uniform(1., SCENE_SIZE) @ "x"
    init_y = uniform(1., SCENE_SIZE) @ "y"

    vx = truncnorm(0., .5, MIN_VELOCITY, MAX_VELOCITY) @ "vx"
    vy = truncnorm(0., .5, MIN_VELOCITY, MAX_VELOCITY) @ "vy"

    blink_rate = normal(BLINK_MEAN, BLINK_STD) @ "blink_rate"
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
def prior_init_fireflies(possible_fireflies):
    """
    Args:
        possible_fireflies: jnp.arange(max_fireflies)
    """
    n_fireflies = labcat(unicat(possible_fireflies), possible_fireflies) @ "n_fireflies"
    masks = jnp.array(possible_fireflies <= n_fireflies)
    max_fireflies = jnp.max(possible_fireflies)
    init_states = init_firefly_at_random.mask().vmap(in_axes=(0))(masks) @ "init"
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
    new_vx = normal(prev_vx, VELOCITY_STD) @ "vx"
    new_vy = normal(prev_vy, VELOCITY_STD) @ "vy"

    # Switch direction on collision
    new_vx = jnp.where((prev_x + new_vx >= SCENE_SIZE - 1.) | (prev_x + new_vx <= 1.), -new_vx, new_vx)
    new_vy = jnp.where((prev_y + new_vy >= SCENE_SIZE - 1.) | (prev_y + new_vy <= 1.), -new_vy, new_vy)

    # Clip new position inside scene
    new_x = jnp.clip(prev_x + new_vx, 0., SCENE_SIZE)
    new_y = jnp.clip(prev_y + new_vy, 0., SCENE_SIZE)

    # Add some noise
    new_x = truncnorm(new_x, POSITION_STD, 0., SCENE_SIZE) @ "x" 
    new_y = truncnorm(new_y, POSITION_STD, 0., SCENE_SIZE) @ "y"

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

@gen
def masked_prior_dynamics(states):
    """
    States is an (n_fireflies,) array of dicts
    masks is an (n_fireflies,) array of mask vals
    obs_x and obs_y are (n_fireflies,) vectors of observations
    """
    masks = states.flag
    model_fn = prior_dynamics_step.mask().vmap(in_axes=(0, 0))
    new_states = model_fn(masks, states.value) @ "steps"
    n_fireflies = jnp.sum(masks)
    
    possible_assignments = jnp.array(list(itertools.permutations(jnp.arange(n_fireflies)))) # scales poorly
    assignment_index = UniformCategorical()(jnp.arange(len(possible_assignments))) @ "assignments"

    assignments = possible_assignments[assignment_index]
    new_states[:n_fireflies] = new_states[assignments]

    return new_states