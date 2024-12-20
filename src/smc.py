import jax
import jax.numpy as jnp
from genjax import ChoiceMapBuilder as C
from utils import get_masked_values, get_observations
from tqdm import tqdm

def maybe_resample(key, log_weights, ess_threshold):
    log_total_weight = jax.nn.logsumexp(log_weights)
    log_normalized_weights = log_weights - log_total_weight
    log_ess = - jax.nn.logsumexp(2 * log_normalized_weights)
    resampled_indices = jax.random.categorical(key, log_normalized_weights, shape=(len(log_weights),))

    ess = jnp.exp(log_ess)
    do_resample = ess < ess_threshold
    particle_inds = (do_resample * resampled_indices) + ((1 - do_resample) * jnp.arange(len(log_weights)))
    return particle_inds, log_total_weight, log_normalized_weights, do_resample, ess

def prop_chm_to_model_chm(prop_chm):
    model_chm = C.n()
    model_chm = C["steps", "blinking"].set(prop_chm["steps", "blinking"])
    model_chm = C["steps", "x"].set(prop_chm["steps", "x"]) | model_chm
    model_chm = C["steps", "y"].set(prop_chm["steps", "y"]) | model_chm
    model_chm = C["steps", "vx"].set(prop_chm["steps", "vx"]) | model_chm
    model_chm = C["steps", "vy"].set(prop_chm["steps", "vy"]) | model_chm
    
    return model_chm

def construct_obs_chm(observed_xs, observed_ys):
    obs_chm = C["observed_xs"].set(observed_xs)
    obs_chm = C["observed_ys"].set(observed_ys) | obs_chm
    return obs_chm

def get_obs_input(model_chm):
    xs = get_masked_values(model_chm["steps", "x"])
    ys = get_masked_values(model_chm["steps", "y"])
    blinks = model_chm["steps", "blinking"].value
    return xs, ys, blinks


def get_obs_input_from_states(states):
    xs = jnp.where(states.flag, states.value["x"], -10.)
    ys = jnp.where(states.flag, states.value["y"], -10.)
    blinks = states.value["blinking"]
    return xs, ys, blinks


def run_particle_filter(gt_trace, gen_fns, n_particles, keygen, ess_threshold=50):
    """
    Args:
        gt_trace: genjax Trace from ground truth model
        gen_fns: tuple of (initialization prior, initialization proposal, dynamics prior, dynamics proposal, likelihood)
        n_particles: int: number of particles
        keygen: JaxKeyHandler object to generate keys
    Returns:
        metrics: dict of states and weights over time
    """

    init_prior, init_proposal, dynamics_prior, dynamics_proposal, likelihood = gen_fns
    max_fireflies, run_steps = gt_trace.get_args()
    gt_chm = gt_trace.get_choices()
    all_x_obs, all_y_obs = get_observations(gt_chm)
    states_over_time = []
    weights_over_time = []
    resamples_over_time = []
    likelihood_over_time = []
    ess_over_time = []
    model_scores_over_time = []
    proposal_scores_over_time = []
    unweighted_posteriors_over_time = []
    obs_x0 = all_x_obs[0]
    obs_y0 = all_y_obs[0]

    keys = keygen(n_particles)
    model_init_traces, model_scores = init_prior(keys, C.n(), (max_fireflies,)) 
    model_init_chms, model_states = jax.vmap(lambda tr: (tr.get_choices(), tr.get_retval()))(model_init_traces)

    keys = keygen(n_particles)
    prop_chms, prop_scores, prop_states = init_proposal(keys, (max_fireflies, obs_x0, obs_y0))
    
    # TODO: Render first state and score
    current_state = prop_states
    # states_over_time.append(current_state)
    for step in tqdm(range(len(run_steps))):
        prev_state = current_state
        obs_x = all_x_obs[step] 
        obs_y = all_y_obs[step] 

        obs_x_broadcast = jnp.tile(obs_x, (n_particles, 1))
        obs_y_broadcast = jnp.tile(obs_y, (n_particles, 1))
        
        # Propose Q(x_t | x_t-1, y_t)
        proposed_constraints, prop_scores, prop_states = dynamics_proposal(keygen(n_particles), (prev_state, obs_x_broadcast, obs_y_broadcast))
        
        # Assess proposed values under prior ( P(x_t | x_{t-1}) )
        translated_prop_chms = jax.vmap(lambda ch: prop_chm_to_model_chm(ch))(proposed_constraints)
        model_scores, model_states = dynamics_prior(translated_prop_chms, (prev_state,))

        # Compute likelihood ( P(y_t | x_t) )        
        obs_chm = construct_obs_chm(obs_x, obs_y)
        obs_input = jax.vmap(lambda t: get_obs_input_from_states(t))(model_states)
        likelihood_traces, likelihood_scores = likelihood(keygen(n_particles), obs_chm, (obs_input))
        
        # Compute weights and possibly resample
        importance = model_scores - prop_scores
        weights = importance + likelihood_scores
        unweighted_posteriors_over_time.append(weights)
        particle_indices, log_total_weight, normalized_weights, resampled, ess = maybe_resample(keygen(), weights, ess_threshold)
        current_state = model_states[particle_indices]
        states_over_time.append(current_state)

        # Track states
        likelihood_over_time.append(likelihood_scores)
        model_scores_over_time.append(model_scores)
        proposal_scores_over_time.append(prop_scores)

        resamples_over_time.append(resampled)
        ess_over_time.append(ess)
        weights_over_time.append(normalized_weights)


    metrics = {
        "unweighted_posteriors": unweighted_posteriors_over_time,
        "prior_weights": model_scores_over_time,
        "proposal_weights": proposal_scores_over_time,
        "likelihoods": likelihood_over_time,
        "ess": ess_over_time,
        "resamples": resamples_over_time,
        "weights": weights_over_time,
    }
    return states_over_time, metrics