using Gen
using Plots
using StatsBase
include("./utilities.jl")
include("./model.jl")
include("./distribution_utils.jl")

function run_mh(particles, i, selection, steps)
    for _ in 1:steps
        particles.traces[i], accepted = mh(particles.traces[i], selection)
    end
end

function mcmc_moves(particles, t, obs)
    # select variables to change: n_fireflies, colors, blink_rates, blinking_states
    num_particles = length(particles.traces)
    scene_size = get_args(particles.traces[1])[1]
    errors = zeros(num_particles, 3, scene_size, scene_size)
    for i in 1:num_particles
        errormap = calc_pixelwise_error(obs, particles.traces[i], t, scene_size, i)
        errors[i, :, :, :] .= errormap
        particle = particles.traces[i]
        choices = get_choices(particle)

        # Vary number of fireflies
        n_fireflies = get_choices(particle)[:init=>:n_fireflies]
        run_mh(particles, i, select(:init => :n_fireflies), 3)
        
        # Vary blink rate
        for n in 1:n_fireflies
            selection = select()
            push!(selection, :init => :blink_rate => n)
            run_mh(particles, i, selection, 2)
        end
        
        # Vary locations
        for n in 1:n_fireflies
            selection = select()
            push!(selection, :init => :init_x => n)
            push!(selection, :init => :init_y => n)
            for prev_t in 1:t - 1
                push!(selection, :states => prev_t => :x => n)
                push!(selection, :states => prev_t => :y => n)
            end
            run_mh(particles, i, selection, 10)
        end

        # Vary blinking states
        for n in 1:n_fireflies
            for prev_t in 1:t - 1
                run_mh(particles, i, select(:states => prev_t => :blinking => n), 2)
            end
        end

         # vary color 
         for n in 1:n_fireflies
            selection = select()
            push!(selection, :init => :color => n)
            run_mh(particles, i, selection, 3)

        end
    end 
end


function mcmc_prior_rejuvenation(particles, mcmc_steps)
    num_particles = length(particles.traces)
    for i in 1:num_particles
        particle = particles.traces[i]
        run_mh(particles, i, selectall(), mcmc_steps)
    end
end


function calc_pixelwise_error(obs, trace, steps, scene_size, particle_id)
    choices = get_choices(trace)
    errors = zeros(3, scene_size, scene_size)
    states, observations = get_retval(trace)
    all_obs = zeros(3, scene_size, scene_size)
    all_states = zeros(3, scene_size, scene_size)
    for t in 1:steps
        observed = choices[:observations => t]
        rendered_image = render(states, t, scene_size)
        errormap = clip.(observed .- rendered_image, 0, 1)

        errors[:, :, :] .+= errormap
        all_obs[:, :, :] .+= observed
        all_states[:, :, :] .+= rendered_image
    end
    fig = plot(layout=(1, 3), axis=nothing)
    # avg_err = Statistics.sum(errors; dims=1)[1, :, :, :]
    heatmap!(fig[1], mat_to_img(all_obs), title="observed", axis=nothing)
    heatmap!(fig[2], mat_to_img(all_states), title="rendered", axis=nothing)
    heatmap!(fig[3], mat_to_img(errors), title="summed error steps $steps", axis=nothing)
    savefig(fig, "errors/p$particle_id-step$steps.png")
    return errors
end


@gen function proposal(trace, obs, step)
    scene_size, max_fireflies, max_steps = get_args(trace)
    old_n_fireflies = trace[:init => :n_fireflies]
    n_fireflies = {:init => :n_fireflies} ~ uniform_discrete(max(1, old_n_fireflies - 1), min(old_n_fireflies + 1, max_fireflies))
    println("prev: ", old_n_fireflies, " new: ", n_fireflies)

    errormap = calc_pixelwise_error(obs, trace, step, scene_size, 0) 
    errormap ./= sum(errormap)
    if n_fireflies > old_n_fireflies
        for n in old_n_fireflies:n_fireflies
            x = {:init => :init_x => n} ~ uniform_discrete(1, scene_size - 1)
            y = {:init => :init_y => n} ~ uniform_discrete(1, scene_size - 1)
            for t in 1:step - 1
                x = {:states => t => :x => n} ~ trunc_norm(Float64(x), 1., 1., Float64(scene_size))
                y = {:states => t => :y => n} ~ trunc_norm(Float64(y), 1., 1., Float64(scene_size))
            end
        end
    end
end

function mcmc(particles, obs, steps, mcmc_steps)
    for i in 1:length(particles.traces)
        particles.traces[i], accepted = mh(particles.traces[i], proposal, (obs, steps))    
    end
end