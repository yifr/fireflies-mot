using Gen
using Images
using Distributions
using Plots
include("./distribution_utils.jl")
include("./utilities.jl")

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
const FIREFLY_COLORS::Vector{Tuple{Float64, Float64, Float64}} = [(1., 0.3, 0.3), (0.3, 1., 0.3), (0.3, 0.3, 1.)]

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
    xs = zeros(Float64, n_fireflies, steps)
    ys = zeros(Float64, n_fireflies, steps)
    vxs = ones(Float64, n_fireflies, steps)
    vys = ones(Float64, n_fireflies, steps)
    motion_var_x = 0.1
    motion_var_y = 0.1
    sigma_x = 1.
    sigma_y = 1.
    colors = zeros(Int, n_fireflies)
    blink_rates = zeros(Float64, n_fireflies)
    blinking_states = zeros(Int, n_fireflies, steps)

    for n in 1:n_fireflies
        color = {:color => n} ~ uniform_discrete(1, 3)
        blink_rate = {:blink_rate => n} ~ uniform(0.999, 0.9999)
        colors[n] = color
        blink_rates[n] = blink_rate
        init_x = {:init_x => n} ~ uniform_discrete(1, scene_size - 1)
        init_y = {:init_y => n} ~ uniform_discrete(1, scene_size - 1)
        init_vx = {:init_vx => n} ~ normal(0, 2)
        init_vy = {:init_vy => n} ~ normal(0, 2)
        xs[n, 1] = Float64(init_x)
        ys[n, 1] = Float64(init_y)
        vxs[n, 1] = Float64(init_vx)
        vys[n, 1] = Float64(init_vy)
    end
    
    return (n_fireflies=n_fireflies, xs=xs, ys=ys, vxs=vxs, vys=vys, colors=colors, 
            blink_rates=blink_rates, blinking_states=blinking_states,
            motion_var_x=motion_var_x, motion_var_y=motion_var_y, sigma_x=sigma_x, sigma_y=sigma_y)
end

@gen function update_states(states, step, scene_size)
    xs = states[:xs]
    ys = states[:ys]
    vxs = states[:vxs]
    vys = states[:vys]
    motion_var_x = states[:motion_var_x]
    motion_var_y = states[:motion_var_y]
    n_fireflies = states[:n_fireflies]
    blink_rates = states[:blink_rates]  
    blinking_states = states[:blinking_states]
    for n in 1:n_fireflies
        prev_step = max(1, step - 1)
        prev_x = xs[n, prev_step]
        prev_y = ys[n, prev_step]
        prev_vx = vxs[n, prev_step]
        prev_vy = vys[n, prev_step]
        blink_rate = blink_rates[n]

        # Update velocity
        vx = {:vx => n} ~ trunc_norm(prev_vx, motion_var_x, -3., 3.)
        vy = {:vy => n} ~ trunc_norm(prev_vy, motion_var_y, -3., 3.)
        
        # Firefly should turn around if it hits the wall
        if prev_x + vx > scene_size || prev_x + vx < 1
            vx = -vx
        end
        if prev_y + vy > scene_size || prev_y + vy < 1
            vy = -vy
        end

        # Update position
        x = {:x => n} ~ trunc_norm(prev_x + vx, 0.01, 1., Float64(scene_size))
        y = {:y => n} ~ trunc_norm(prev_y + vy, 0.01, 1., Float64(scene_size))

        # Update blinking
        blinking = {:blinking => n} ~ bernoulli(blink_rate)
        xs[n, step] = x
        ys[n, step] = y
        vxs[n, step] = vx
        vys[n, step] = vy
        blinking_states[n, step] = blinking
    end

    return states
end

function add_firefly_glow!(pixels, x_loc, y_loc, x_sigma, y_sigma, color)
    """
    Modify a pixel map to add the glow of a firefly
    """
    r, g, b = FIREFLY_COLORS[color]
    scene_size = size(pixels)[2]

    # x_factor = -1 / (2 * x_sigma^2)
    # y_factor = -1 / (2 * y_sigma^2)
    
    xmin = max(1, round(Int, x_loc - 2 * x_sigma))
    xmax = min(scene_size, round(Int, x_loc + 2 * x_sigma))
    ymin = max(1, round(Int, y_loc - 2 * y_sigma))
    ymax = min(scene_size, round(Int, y_loc + 2 * y_sigma))

    for i in xmin:xmax
        # dx2 = (i - x_loc)^2
        for j in ymin:ymax
            # dy2 = (j - y_loc)^2
            # alpha = exp(x_factor * dx2 + y_factor * dy2) 
            alpha = 1
            #if alpha > 0.01
            @inbounds pixels[1, j, i] = r * alpha
            @inbounds pixels[2, j, i] = g * alpha
            @inbounds pixels[3, j, i] = b * alpha
            #end
        end
    end

    return pixels
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
            pixels = add_firefly_glow!(pixels, x, y, sigma_x, sigma_y, color)
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


@gen function model(scene_size::Int64, max_fireflies::Int64, steps::Int64)
    """
    Given some scene size, and max number of fireflies, run the model for a number of steps
    """
    states = {:init} ~ initialize_fireflies(scene_size, max_fireflies, steps)
    observations = zeros(Float64, steps, 3, scene_size, scene_size)
    for t in 1:steps
        states = {:states => t} ~ update_states(states, t, scene_size)
        rendered_state = render!(states, t, scene_size)
        observation = {:observations => t} ~ image_likelihood(rendered_state, 0.01)
        observations[t, :, :, :] .= rendered_state
    end
    return states, observations
end