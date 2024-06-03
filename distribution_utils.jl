using Gen
using Distributions

function uniprobs(vec)
	probs = fill(1/length(vec), length(vec))
	return probs
end

@dist function uniform_cat(size::Int)
	probs = fill(1/size, size)
	categorical(probs)
end

@dist function labeled_cat(labels, probs)
	index = categorical(probs)
	labels[index]
end

struct TruncNorm <: Gen.Distribution{Float64} end

const trunc_norm = TruncNorm()

function Gen.logpdf(::TruncNorm, value::Float64, mu::Float64, sigma::Float64, low::Float64, high::Float64)
	d = Truncated(Normal(mu, sigma), low, high)
	return Distributions.logpdf(d, value)
end

function Gen.random(::TruncNorm, mu::Float64, sigma::Float64, low::Float64, high::Float64)
	d = Truncated(Normal(mu, sigma), low, high)
	return rand(d)
end

logpdf_grad(::TruncNorm) = nothing 
has_output_grad(::TruncNorm) = false
has_argument_grads(::TruncNorm) = (false, false, false, false)