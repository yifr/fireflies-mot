using Gen


function add_firefly_glow!(pixels, x_loc, y_loc, x_sigma, y_sigma, color)
    """
    Modify a pixel map to add the glow of a firefly
    """
    r, g, b = FIREFLY_COLORS[color]
    scene_size = size(pixels)[1]

    x_factor = -1 / (3 * x_sigma^2)
    y_factor = -1 / (3 * y_sigma^2)
    
    xmin = round(Int64, max(1, x_loc - 2 * x_sigma))
    xmax = round(Int64, min(scene_size, x_loc + 2 * x_sigma))
    ymin = round(Int64, max(1, y_loc - 2 * y_sigma))
    ymax = round(Int64, min(scene_size, y_loc + 2 * y_sigma))

    for i in xmin:xmax
        dx2 = (i - x_loc)^2
        for j in ymin:ymax
            dy2 = (j - y_loc)^2
            alpha = exp(x_factor * dx2 + y_factor * dy2)
            if alpha > 0.01
                @inbounds pixels[j, i, 1] += alpha * r
                @inbounds pixels[j, i, 2] += alpha * g
                @inbounds pixels[j, i, 3] += alpha * b
            end
        end
    end

    return nothing
end


function render!(states::NamedTuple, step::Int64, scene_size::Int64)
    """
    Deterministic renderer
    """
    xs = states[:xs]
    ys = states[:ys]
    colors = states[:colors]
    n_fireflies = states[:n_fireflies]
    blinking_states = states[:blinking_states]
    sigma_x = states[:sigma_x]
    sigma_y = states[:sigma_y]
    pixels = zeros(Float64, 3, scene_size, scene_size)
    for n in 1:n_fireflies
        x = xs[n, step]
        y = ys[n, step]
        color = colors[n]
        blinking = blinking_states[n, step]
        if blinking == 1
            add_firefly_glow!(pixels, x, y, sigma_x, sigma_y, color)
        end
    end
    
    return pixels
end



struct ImageLikelihood <: Gen.Distribution{Array} end

function Gen.logpdf(::ImageLikelihood, observed_image::Array{Float64,3}, rendered_image::Array{Float64,3}, var)
    # precomputing log(var) and assuming mu=0 both give speedups here
    log_var = log(var)
    sum(i -> - (@inbounds abs2((observed_image[i] - rendered_image[i]) / var) + log(2π)) / 2 - log_var, eachindex(observed_image))
end

function logpdfmap(::ImageLikelihood, observed_image::Array{Float64,3}, rendered_image::Array{Float64,3}, var::Float64)
    # map of logpdf (heatmap) over each pixel, how correct it is
    log_var = log(var)

    C, H, W = size(observed_image)

    heatmap = zeros(Float64, H, W)

    for hi in 1:H
        for wi in 1:W
            for ci in 1:3
                heatmap[hi, wi] += -(@inbounds abs2((observed_image[ci, hi, wi] - rendered_image[ci, hi, wi]) / var) + log(2π)) / 2 - log_var
            end
        end
    end
    heatmap
end 

function Gen.random(::ImageLikelihood, rendered_image::Array{Float64, 3}, var::Float64)
    noise = rand(Distributions.Normal(0, var), size(rendered_image))
    rendered_image .+ noise
end

const image_likelihood = ImageLikelihood()
(::ImageLikelihood)(rendered_image, var) = random(ImageLikelihood(), rendered_image, var)
