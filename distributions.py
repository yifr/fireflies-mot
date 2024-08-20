import jax.numpy as jnp
from genjax import ExactDensity, Pytree
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

unicat = lambda x: jnp.ones(len(x)) / len(x)
normalize = lambda x: x / jnp.sum(x)

@Pytree.dataclass
class LabeledCategorical(ExactDensity):
    def sample(self, key, probs, labels, **kwargs):
        cat = tfd.Categorical(probs=normalize(probs))
        cat_index = cat.sample(seed=key)
        return labels[cat_index]

    def logpdf(self, v, probs, labels, **kwargs):
        w = jnp.log(jnp.sum(normalize(probs) * (labels==v)))
        return w

@Pytree.dataclass
class UniformCategorical(ExactDensity):
    def sample(self, key, labels, **kwargs):
        cat = tfd.Categorical(probs=jnp.ones(len(labels)) / len(labels))
        cat_index = cat.sample(seed=key)
        return labels[cat_index]

    def logpdf(self, v, labels, **kwargs):
        probs = jnp.ones(len(labels)) / len(labels)
        logpdf = jnp.log(probs)
        w = logpdf[0]
        return w
    
labcat = LabeledCategorical()
uniformcat = UniformCategorical()