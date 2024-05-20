using Gen

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


function observe_at_time(trace, step)
    args = get_args(trace)
    scene_size = args[1]
    pixels = zeros(Int, scene_size, scene_size)
    choices = get_choices(trace)
    for x in 1:scene_size
        for y in 1:scene_size
            pixels[x, y] = choices[(:pixels, x, y, step)]
        end
    end
    
    return pixels
end

"""
The default proposal doesn't work under the current model because of how fireflies are indexed.
If a particle's originally estimates :n_fireflies = 1, and then gets an observation indexing a firefly with n > 1,
the update step will be unable to constrain the observation of (:pos, n>1, t) and will fail.
"""
@gen function particle_filter_default_proposal(trace, model, num_particles::Int, num_samples::Int)
    scene_size, steps, max_fireflies = get_args(trace)
    init_obs = get_traced_variable_observation(trace, step=1)
    state = Gen.initialize_particle_filter(model, (scene_size, 1, max_fireflies,), init_obs, num_particles)
    
    println(init_obs)
    for t=2:steps
        Gen.maybe_resample!(state, ess_threshold=num_particles/2)
        obs = get_traced_variable_observation(trace, step=t)
        println("\n\n(T=$t) Observations: \n$obs\n")
        Gen.particle_filter_step!(state, (scene_size, t, max_fireflies,), (NoChange(), UnknownChange(), NoChange(),), obs)
    end
    
    # return a sample of unweighted traces from the weighted collection
    return Gen.sample_unweighted_traces(state, num_samples)
end;


function particle_filter_rejuv_resim(trace, model, num_particles::Int, num_samples::Int)
    scene_size, steps, max_fireflies = get_args(trace)
    init_obs = get_traced_variable_observation(trace, step=1)
    state = Gen.initialize_particle_filter(model, (scene_size, 1, max_fireflies,), init_obs, num_particles)

    for t=2:steps
        obs = get_traced_variable_observation(trace, step=t)

        # apply a rejuvenation move to each particle
        for i=1:num_particles
            initial_choices = select(:x0, :y0, :vx0, :vy0)
            state.traces[i], _  = mh(state.traces[i], initial_choices)
        end

        Gen.maybe_resample!(state, ess_threshold=num_particles/2)
        
        Gen.particle_filter_step!(state, (t,), (UnknownChange(),), obs)
    end

    # return a sample of unweighted traces from the weighted collection
    return Gen.sample_unweighted_traces(state, num_samples)
end;
