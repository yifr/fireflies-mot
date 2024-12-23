import genjax
from genjax import gen

import jax
import jax.numpy as jnp


@gen 
def noisy_position_likelihood(xs, ys, blinks):
    """
    Observes all "blinking" states with some positional noise
    """
    observed_xs = jnp.full_like(xs, -10.)
    observed_ys = jnp.full_like(ys, -10.)
    observed_xs = jnp.where(blinks, xs, observed_xs)
    observed_ys = jnp.where(blinks, ys, observed_ys)
    
    observed_xs = genjax.normal(observed_xs, .25) @ "observed_xs"
    observed_ys = genjax.normal(observed_ys, .25) @ "observed_ys"

    return jnp.stack([observed_xs, observed_ys])