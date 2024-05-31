using Gen
using Images
using Distributions
using Plots
include("./distribution_utils.jl")

"""
Firefly model:

initialize (scene_size, max_fireflies)
 - places fireflies in a scene and initializes colors and blink rates

update_states (scene_size, states)
 - for each firefly, random walk position and blinking rates
 - at some point this can be made to include more complex dynamics / behaviors

render 
    - deterministic function that maps latent states to pixels

observe
    - Defines a noisy likelihood for observations

model(max_fireflies, steps): 
    - returns trace and observations

"""


@gen function initialize_fireflies(scene_size::Int64, max_fireflies::Int64, steps::Int64)
    """
    Initializes a random number of fireflies with x, y, color and blink rates

    Traced variables:
    - n_fireflies: number of fireflies
    - (x, n, t): x position of firefly n at time t
    - (y, n, t): y position of firefly n at time t
    - (color, n): color of firefly n
    - (blink_rate, n): frequency of blinking for firefly n
    """
    n_fireflies = {:n_fireflies} ~ uniform_discrete(1, max_fireflies)
    xs = zeros(Float64, n_fireflies, steps + 1)
    ys = zeros(Float64, n_fireflies, steps + 1)
    sigma_x = 1
    sigma_y = 2
    motion_var_x = 1
    motion_var_y = 1
    colors = zeros(Int, n_fireflies)
    blink_rates = zeros(Float64, n_fireflies)
    blinking_states = zeros(Int, n_fireflies, steps + 1)
    t = 1
    for n in 1:n_fireflies
        color = {:color => n} ~ uniform_discrete(1, 3)
        blink_rate = {:blink_rate => n} ~ uniform(0.1, 0.5)
        colors[n] = color
        blink_rates[n] = blink_rate
    end
    
    return (n_fireflies=n_fireflies, xs=xs, ys=ys, colors=colors, blink_rates=blink_rates, blinking_states=blinking_states,
            sigma_x=sigma_x, sigma_y=sigma_y, motion_var_x=motion_var_x, motion_var_y=motion_var_y)
end

@gen function update_states(states, step, scene_size)
    xs = states[:xs]
    ys = states[:ys]
    motion_var_x = states[:motion_var_x]
    motion_var_y = states[:motion_var_y]
    n_fireflies = states[:n_fireflies]
    blink_rates = states[:blink_rates]  
    blinking_states = states[:blinking_states]
    for n in 1:n_fireflies
        if step == 1
            prev_x = uniform_discrete(1, scene_size - 1)
            prev_y = uniform_discrete(1, scene_size - 1)
        else
            prev_x = xs[n, step-1]
            prev_y = ys[n, step-1]
        end
        blink_rate = blink_rates[n]
        x = {:x => n => step} ~ normal(prev_x, motion_var_x)
        y = {:y => n => step} ~ normal(prev_y, motion_var_y)
        blinking = {:blinking => n => step} ~ bernoulli(blink_rate)
        xs[n, step] = x
        ys[n, step] = y
        blinking_states[n, step] = blinking
    end

    return states
end

function calculate_firefly_glow(x_loc, y_loc, x_sigma, y_sigma, scene_size)
    """
    For each firefly, calculate glow for each pixel in the scene
    """
    alphas = zeros(Float64, scene_size, scene_size)
    for row in 1:scene_size
        for col in 1:scene_size
            alpha = exp(-((col - x_loc)^2 / (2 * x_sigma^2) + (row - y_loc)^2 / (2 * y_sigma^2)))
            if alpha > 0.01
                alphas[row, col] = alpha
            end
        end
    end
    return alphas
end

function mat_to_img(mat)
    """
    Convert matrix to image
    """
    img = colorview(RGBA, clip.(mat, 0., 1.))
    return img
end

function render(states, step::Int64, scene_size::Int64)
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
    color_map = [(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)]
    pixels = zeros(Float64, 4, scene_size, scene_size)
    for n in 1:n_fireflies
        x = xs[n, step]
        y = ys[n, step]
        color = colors[n]
        blinking = blinking_states[n, step]
        if blinking == 1
            alphas = calculate_firefly_glow(x, y, sigma_x, sigma_y, scene_size)            
            r, g, b = color_map[color]
            pixels[1, :, :] .+= alphas .* r
            pixels[2, :, :] .+= alphas .* g
            pixels[3, :, :] .+= alphas .* b
            pixels[4, :, :] .+= alphas
        end
    end
    # img = clip.(pixels, 0., 1.)
    # img = colorview(RGBA, img)
    # display(heatmap(img, xlims=(0, scene_size), ylims=(0, scene_size), 
    #     aspect_ratio=1,legend=false, background_color=:black))

    return pixels
end


struct ImageLikelihood <: Gen.Distribution{Array} end

function Gen.logpdf(::ImageLikelihood, observed_image::Array{Float64,3}, rendered_image::Array{Float64,3}, var)
    # precomputing log(var) and assuming mu=0 both give speedups here
    log_var = log(var)
    sum(i -> - (@inbounds abs2((observed_image[i] - rendered_image[i]) / var) + log(2π)) / 2 - log_var, eachindex(observed_image))
end

function logpdfmap(::ImageLikelihood, observed_image::Array{Float64,3}, rendered_image::Array{Float64,3}, var)
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

function Gen.random(::ImageLikelihood, rendered_image, var)
    noise = rand(Distributions.Normal(0, var), size(rendered_image))
    rendered_image .+ noise
end

const image_likelihood = ImageLikelihood()
(::ImageLikelihood)(rendered_image, var) = random(ImageLikelihood(), rendered_image, var)


@gen function model(scene_size::Int64, max_fireflies::Int64, steps::Int64)
    """
    Given some scene size, and max number of fireflies, run the model for a number of steps
    """
    states = {:init} ~ initialize_fireflies(scene_size, max_fireflies, steps)
    observations = zeros(Float64, steps, 4, scene_size, scene_size)
    for t in 1:steps
        states = {:states => t} ~ update_states(states, t, scene_size)
        rendered_state = render(states, t, scene_size)
        observation = {:observations => t} ~ image_likelihood(rendered_state, 0.01)
        observations[t, :, :, :] = observation
    end
    return states, observations
end
