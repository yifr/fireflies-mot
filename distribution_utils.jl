using Gen

function uniprobs(vec)
	probs = fill(1/length(vec), length(vec))
	return probs
end

# TODO: fix this
@dist function truncated_normal(mu, sigma, lower, upper)
	normal(mu, sigma)
end

@dist function uniform_cat(size::Int)
	probs = fill(1/size, size)
	categorical(probs)
end

@dist function labeled_cat(labels, probs)
	index = categorical(probs)
	labels[index]
end
