using Gen
using JSON

include("./visualizations.jl")
include("./utilities.jl")

function get_traced_variable_observation(trace; step)
    args = get_args(trace)
    choices = get_choices(trace)
    n_fireflies = choices[:n_fireflies]

    # if end_step == -1
    #     total_steps = args[2]
    #     steps = 1:total_steps
    # else
    #     steps = 1:end_step
    # end

    observations = []
    chm = Gen.choicemap()

    for n in 1:n_fireflies
        blinking = choices[(:blinking, n, step)]
        if blinking == 1
            x = choices[(:x, n, step)]
            y = choices[(:y, n, step)]
            color = choices[(:color, n)]
            chm[(:x, n, step)] = Int(trunc(x))
            chm[(:y, n, step)] = Int(trunc(y))
            chm[(:blinking, n, step)] = 1
        
            # If you haven't seen this firefly before, log the color
            first_obs = true
            for t in 1:step
                if choices[(:blinking, n, t)] == 1 && t < step
                    first_obs = false
                    break
                end
            end
            # if first_obs
            #     chm[(:color, n)] = color
            # end
        end
    end

    return chm
end

function clip(num, low, hi)
    return max(low, min(num, hi))
end

function observe_at_time(trace, step)
    args = get_args(trace)
    scene_size = args[1]
    pixels = zeros(Int, scene_size, scene_size)
    choices = get_choices(trace)
    #var = 0.001
    #noise = normal.(zeros(scene_size, scene_size), var)
    chm = choicemap()
    for x in 1:scene_size
        for y in 1:scene_size
            chm[(:pixels, step, x, y)] = clip(choices[(:pixels, step, x, y)], 0., 1.)
        end
    end
    
    return chm
end


"""
The default proposal doesn't work under the current model because of how fireflies are indexed.
If a particle's originally estimates :n_fireflies = 1, and then gets an observation indexing a firefly with n > 1,
the update step will be unable to constrain the observation of (:pos, n>1, t) and will fail.
"""
@gen function particle_filter_default_proposal(trace, model, num_particles::Int, num_samples::Int)
    scene_size, steps, max_fireflies = get_args(trace)
    init_obs = observe_at_time(trace, 1)
    state = Gen.initialize_particle_filter(model, (scene_size, 1, max_fireflies,), init_obs, num_particles)
    
    for t=2:steps
        println("t=$t")
        Gen.maybe_resample!(state, ess_threshold=num_particles/2)
        obs = observe_at_time(trace, t)
        Gen.particle_filter_step!(state, (scene_size, t, max_fireflies,), (NoChange(), UnknownChange(), NoChange(),), obs)
    end

    # return a sample of unweighted traces from the weighted collection
    try 
        println("Success")
        return Gen.sample_unweighted_traces(state, num_samples)
    catch e
        println("Failed")
        return state.traces[1:5]
    end
end;

function mean(arr)
    return sum(arr) / length(arr)
end

function make_record(particle, savedir, t, i)
    record = Dict()
    particle_state, _ = get_retval(particle)
    scene_size, _, _ = get_args(particle)
    rendered_img = mat_to_img(render(particle_state, t, scene_size))
    record["state"] = particle_state
    record["score"] = get_score(particle)

    # save rendered_img to file
    save_path = joinpath(savedir, "images", "t$t-particle$i.png")
    mkpath(dirname(save_path))
    
    record["rendered_img"] = save_path
    save(save_path, rendered_img)
    return record
end

function smc(trace, model, num_particles::Int, num_samples::Int; record_json=true, experiment_tag="")
    scene_size, max_fireflies, steps = get_args(trace)

    obs = get_choices(trace)[:observations => 1]
    chm = choicemap()
    chm[:observations=>1] = obs
    
    state = Gen.initialize_particle_filter(model, (scene_size, max_fireflies,1), chm, num_particles)
    if record_json
        savedir = timestamp_dir(experiment_tag=experiment_tag)
        res_json = Dict(
            "num_particles" => num_particles,
            "num_samples" => num_samples,
            "experiment_tag" => experiment_tag,
            "scene_size" => scene_size,
            "max_fireflies" => max_fireflies,
            "steps" => steps,
            "smc_steps" => [[Dict() for _ in 1:num_particles] for _ in 1:steps]
        )
        for i=1:num_particles
            t = 1
            particle = state.traces[i]
            record = make_record(particle, savedir, t, i)
            res_json["smc_steps"][t][i] = record
        end
    end

    mh_accepted = []
    for t=2:steps
        if record_json
            particles = [state.traces[i] for i in 1:num_particles]
            anim = visualize_particles(particles, trace)
            save_path = joinpath(savedir, "videos", "t$t-particles.mp4")
            mkpath(dirname(save_path))
            mp4(anim, save_path, fps=5)
        end

        # write out observation and save filepath name
        # record all the particle traces at time t - 1
        println()
        println("t=$t")
        
        # apply a rejuvenation move to each particle
        num_accepted = 0
        for i=1:num_particles
            # select variables to change: n_fireflies, colors, blink_rates, blinking_states
            particle = state.traces[i]
            n_fireflies = get_choices(particle)[:init=>:n_fireflies]
            choices = select(:init => :n_fireflies)
            for n in 1:n_fireflies
                push!(choices, :init => :color => n)
                push!(choices, :init => :blink_rate => n)
                push!(choices, :states => t => :blinking => n)
                push!(choices, :states => t => :x => n)
                push!(choices, :states => t => :y => n)
            end
            state.traces[i], accepted  = mh(state.traces[i], choices)
            num_accepted += accepted
        end 
        push!(mh_accepted, num_accepted / num_particles)
        obs = get_choices(trace)[:observations => t]
        chm = choicemap()
        chm[:observations => t] = obs

        Gen.maybe_resample!(state, ess_threshold=num_particles/2)
        Gen.particle_filter_step!(state, (scene_size, max_fireflies, t,), (NoChange(), UnknownChange(), NoChange(),), chm)
        if record_json
            for i=1:num_particles
                particle = state.traces[i]
                res_json["smc_steps"][t-1][i] = make_record(particle, savedir, t-1, i)
            end
        end
    end

    if record_json
        save_path = joinpath(savedir, "results.json")
        open(save_path, "w") do f
            JSON.print(f, res_json)
        end
    end

    display(plot([mh_accepted], title="MH Acceptance Rate", xlabel="Step", ylabel="Acceptance Rate"))
    # return a sample of unweighted traces from the weighted collection
    return Gen.sample_unweighted_traces(state, num_samples)
end;
