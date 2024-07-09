using Gen
using Plots
using StatsBase
using Combinatorics
include("./utilities.jl")
# include("./model.jl")
include("./distribution_utils.jl")

function run_mh(particles::Gen.ParticleFilterState, i::Int64, selection::DynamicSelection, steps::Int64, label=Val{:unlabeled}())
    num_accepted = 0
    for _ in 1:steps
        particles.traces[i], accepted = mh(particles.traces[i], selection)
        num_accepted += accepted
    end
    return num_accepted / steps
end

function get_prev_blink(trace::Gen.DynamicDSLTrace, n::Int64, current_t::Int64)
    """
    Return time step of previous blink for firefly n, or 1 if no previous blink
    """
    choices = get_choices(trace)
    for t in current_t - 1:-1:1
        if choices[:states => t => :blinking => n] == 1
            return t
        end
    end
    
    return return 1
end

function get_all_combinations(lst)
    result = []
    for i in 1:length(lst)
        append!(result, collect(combinations(lst, i)))
    end
    return result
end

function mh_block_rejuvenation(particles::Gen.ParticleFilterState, t::Int64, obs::Array{Float64, 3})
    # select variables to change: n_fireflies, colors, blink_rates, blinking_states
    num_particles = length(particles.traces)
    scene_size = get_args(particles.traces[1])[1]

    nf_accepted = 0 # stats for num fireflies accepted
    location_accepted = 0 # stats for location changes accepted
    blinking_accepted = 0 # stats for blinking changes accepted
    color_accepted = 0 # stats for color changes accepted
    for i in 1:num_particles
        particle = particles.traces[i]
        choices = get_choices(particle)

        # Vary number of fireflies
        n_fireflies = get_choices(particle)[:init=>:n_fireflies]
        accepted = run_mh(particles, i, select(:init => :n_fireflies), 100, Val{:n_fireflies}())
        nf_accepted += accepted

        # Vary locations from previous blink - quasi counterfactual that says "could this firefly have ended up here"
        for n in 1:n_fireflies
            selection = select()
            prev_blink = get_prev_blink(particle, n, t)
            for prev_t in prev_blink : t
                push!(selection, :states => prev_t => :x => n)
                push!(selection, :states => prev_t => :y => n)
            end
            push!(selection, :states => t => :blinking => n)
            accepted = run_mh(particles, i, selection, 100, Val{:trajectory}())
            location_accepted += accepted
        end

        for n in 1:n_fireflies
            selection = select()
            push!(selection, :states => t => :blinking => n)
            accepted = run_mh(particles, i, selection, 100, Val{:blink}())
            blinking_accepted += accepted
        end

        # vary color 
        for n in 1:n_fireflies
            selection = select()
            push!(selection, :init => :color => n)
            accepted = run_mh(particles, i, selection, 4, Val{:color}())
            color_accepted += accepted
        end
    end 
    
    nf_accepted = nf_accepted / num_particles
    location_accepted = location_accepted / num_particles
    blinking_accepted = blinking_accepted / num_particles
    color_accepted = color_accepted / num_particles

    println("NF: ", nf_accepted, " Location: ", location_accepted, " Blinking: ", blinking_accepted, " Color: ", color_accepted, "\n")
    # return nf_accepted, location_accepted, blinking_accepted, color_accepted
end


function mcmc_prior_rejuvenation(particles::Gen.ParticleFilterState, mcmc_steps::Int64)
    num_particles = length(particles.traces)
    for i in 1:num_particles
        particle = particles.traces[i]
        run_mh(particles, i, selectall(), mcmc_steps)
    end
end


function average_reconstruction_error(trace::Gen.DynamicDSLTrace)
    scene_size, _, steps = get_args(trace)
    choices = get_choices(trace)
    errors = zeros(3, scene_size, scene_size)
    states, observations = get_retval(trace)
    for t in 1:steps
        observed = choices[:observations => t]
        rendered_image = render(states, t, scene_size)
        errormap = Base.clamp.(observed .- rendered_image, 0, 1)
        errors .+= errormap
    end

    # Average over time and color
    errors = errors ./ steps
    return errors
end

function find_low_likelihood_regions(errors::Array{Float64, 3})
    """
    Given a (3, H, W) image of average reconstruction errors
    return distribution of x and y locations with most common errors
    """
    errors = StatsBase.mean(errors; dims=[1,])[1, :, :] # shape: (scene_size, scene_size)
    if sum(errors) == 0
        return [0], [0]
    else
        # Normalize to distribution over pixels
        x_errors = StatsBase.mean(errors; dims=2)[:, 1] # (scene_size,)
        x_errors ./= sum(x_errors)
        y_errors = StatsBase.mean(errors; dims=1)[1, :] # (scene_size,)
        y_errors ./= sum(y_errors)
        return x_errors, y_errors 
    end
end


function observed_color_hist(trace)
    scene_size, _, steps = get_args(trace)
    choices = get_choices(trace)
    all_obs = zeros(3, scene_size, scene_size)
    for t in 1:steps
        all_obs .+= choices[:observations => t]
    end
    color_hist = Base.clamp.(StatsBase.mean(all_obs; dims=[2,3])[:, 1, 1], 0., 1.) # shape: (3,)
    color_hist = color_hist / sum(color_hist)
    return color_hist
end

@gen function proposal(trace::Gen.DynamicDSLTrace)
    scene_size, max_fireflies, max_steps = get_args(trace)
    old_n_fireflies = trace[:init => :n_fireflies]
    n_fireflies = {:init => :n_fireflies} ~ uniform_discrete(max(1, old_n_fireflies - 1), min(old_n_fireflies + 1, max_fireflies))
    states, _ = get_retval(trace)

    recon_errors = average_reconstruction_error(trace)
    low_likelihood_x, low_likelihood_y = find_low_likelihood_regions(recon_errors) 
    color_hist = observed_color_hist(trace)

    if n_fireflies > old_n_fireflies
        n = n_fireflies
        if sum(low_likelihood_x) == 1
            x = {:init => :init_x => n} ~ categorical(low_likelihood_x) 
            y = {:init => :init_y => n} ~ categorical(low_likelihood_y) 
        else
            x = {:init => :init_x => n} ~ uniform_discrete(1, scene_size - 1) 
            y = {:init => :init_y => n} ~ uniform_discrete(1, scene_size - 1) 
        end
        
        if sum(color_hist) == 1
            color = {:init => :color => n} ~ categorical(color_hist)
        else
            color = {:init => :color => n} ~ uniform_discrete(1, 3)
        end

        blink_rate = {:init => :blink_rate => n} ~ uniform(0.1, 0.25)
        for t in 1:max_steps
            # upweight blinking in low likelihood areas
            x = {:states => t => :x => n} ~ trunc_norm(Float64(x), 1., 1., Float64(scene_size))
            y = {:states => t => :y => n} ~ trunc_norm(Float64(y), 1., 1., Float64(scene_size))
            upweight_prob = 0
            if sum(low_likelihood_x) == 1
                upweight_x = categorical(low_likelihood_x) 
                upweight_y = categorical(low_likelihood_y) 
                x_prob = Gen.logpdf(normal, x, Float64(upweight_x), 3.)
                y_prob = Gen.logpdf(normal, y, Float64(upweight_y), 3.)
                upweight_prob = exp(x_prob + y_prob)
            else
                upweight_prob = 0
            end
            blinking = {:states => t => :blinking => n} ~ bernoulli(blink_rate + upweight_prob)     
        end
    end
    nothing
end

function data_driven_mcmc(particles, obs, steps, mcmc_steps)
    for i in 1:length(particles.traces)
        for _ in 1:mcmc_steps
            particles.traces[i], accepted = mh(particles.traces[i], proposal, ())    
        end
    end
end