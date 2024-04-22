import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import genjax
from genjax import GenerativeFunction, ChoiceMap, JAXGenerativeFunction, ExactDensity, Selection, trace
from genjax.generative_functions.distributions import TFPDistribution
from genjax.typing import Callable 
from dataclasses import dataclass
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

""" Define Distributions """
    
# @dataclass
class LabeledCategorical(JAXGenerativeFunction, ExactDensity):
    def sample(self, key, probs, labels, **kwargs):
        cat = tfd.Categorical(probs=probs)
        cat_index = cat.sample(seed=key)
        return labels[cat_index]

    def logpdf(self, v, probs, labels, **kwargs):
        w = jnp.log(jnp.sum(probs * (labels==v)))
        return w

class UniformCategorical(JAXGenerativeFunction, ExactDensity):
    def sample(self, key, labels, **kwargs):
        cat = tfd.Categorical(probs=jnp.ones(len(labels)) / len(labels))
        cat_index = cat.sample(seed=key)
        return labels[cat_index]

    def logpdf(self, v, labels, **kwargs):
        probs = jnp.ones(len(labels)) / len(labels)
        logpdf = jnp.log(probs)
        w = logpdf[0]
        return w
    

def discrete_norm(μ, σ, dom):
    return normalize(
        jax.vmap(
            lambda i: tfd.Normal(loc=μ, scale=σ).cdf(i + .5) - tfd.Normal(loc=μ, scale=σ).cdf(i - .5))(dom))

def truncate(μ, dom, winsize, normvals):
    return (jnp.abs(dom-μ) <= winsize) * normvals

# enter a negative winsize for going the other direction. 
def upweight_zone(winstart, dom, winsize, density_in_win):
    weight_in_win = density_in_win / jnp.abs(winsize)
    weight_outside_win = (1-density_in_win) / (len(dom) - jnp.abs(winsize))
    upweighted_window = (((dom - winstart) * jnp.sign(winsize)) >= 0) & (((dom - winstart) * jnp.sign(winsize)) < jnp.abs(winsize))
    return weight_in_win * upweighted_window + weight_outside_win * ~upweighted_window


def sample_test(genfn, probs, labels):
    global key
    key, subkey = jax.random.split(key, 2)
    sample = genfn.simulate(subkey, (probs, labels))
    return sample.get_retval()


cat = TFPDistribution(lambda p: tfd.Categorical(probs=p))
labcat = LabeledCategorical()
uniformcat = UniformCategorical()

unicat_probs = lambda length: jnp.ones(length) / length
normalize = lambda x: x / jnp.sum(x)

discrete_truncnorm = lambda μ, σ, winsize, dom: normalize(truncate(μ, dom, winsize, discrete_norm(μ, σ, dom)))
onehot = lambda x, dom: discrete_truncnorm(x , 1, 0, dom)

from dataclasses import dataclass

class UniformDiscreteArray(ExactDensity, JAXGenerativeFunction):
    def sample(self, key, vals, arr):
        return jax.random.choice(key, vals, shape=arr.shape)

    def logpdf(self, sampled_val, vals, arr, **kwargs):
        return jnp.log(1.0 / (vals.shape[0])) * arr.shape[0]

class UniformChoice(ExactDensity, JAXGenerativeFunction):
    def sample(self, key, vals):
        return jax.random.choice(key, vals)

    def logpdf(self, sampled_val, vals, **kwargs):
        valid = jnp.isin(sampled_val, vals)
        log_probs = jnp.where(valid, -jnp.log(vals.shape[0]), -jnp.inf)
        return log_probs

class UniformDiscrete(ExactDensity, JAXGenerativeFunction):
    """
    uniform_discrete(a, b) samples a uniform integer x such that a <= x < b.
    If a is not less than b, the result is always a.
    """
    def sample(self, key, low, high):
        return jax.random.randint(key, shape=(), minval=low, maxval=high)

    def logpdf(self, sampled_val, low, high, **kwargs):
        range_is_nontrivial = low + 1 <= high
        equals_low = low == sampled_val
    
        is_in_range = (low <= sampled_val) & (sampled_val < high)

        log_probs_branch1 = jnp.where(equals_low, 0., -jnp.inf)
        log_probs_branch2 = jnp.where(is_in_range, -jnp.log(high - low), -jnp.inf)
        log_probs = jnp.where(range_is_nontrivial, log_probs_branch2, log_probs_branch1)

        return log_probs

uniform_discrete = UniformDiscrete()
uniform_choice = UniformChoice()
uniform_discrete_array = UniformDiscreteArray()
poisson = TFPDistribution(tfd.Poisson)