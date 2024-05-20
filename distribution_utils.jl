using Gen

function uniprobs(vec)
	probs = fill(1/length(vec), length(vec))
	return probs
end

# TODO: fix this
@dist function truncated_normal(mu, sigma, lower, upper)
	normal(mu, sigma)
end

# @dist binned_normal_2d(mu_x, mu_y, sigma_x, sigma_y, bins)
# 	x = normal(mu_x, sigma_x)
# 	y = normal(mu_y, sigma_y)
# 	bin_x = Int(trunc(x))
# 	bin_y = Int(trunc(y))
# 	(bin_x, bin_y)
# end

@dist function uniform_cat(size::Int)
	probs = fill(1/size, size)
	categorical(probs)
end

@dist function labeled_cat(labels, probs)
	index = categorical(probs)
	labels[index]
end
