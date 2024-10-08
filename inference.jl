using Gen
using JSON
include("./mcmc.jl")
include("./visualizations.jl")
include("./utilities.jl")

function make_record(particle::Gen.DynamicDSLTrace, savedir::String, t::Int64, i::Int64)
    record = Dict()
    particle_state, _ = get_retval(particle)
    scene_size, _, _ = get_args(particle)
    rendered_img = mat_to_img(render!(particle_state, t, scene_size))
    record["state"] = particle_state
    record["score"] = get_score(particle)
    # save rendered_img to file
    save_path = joinpath(savedir, "images", "t$t-particle$i.png")
    mkpath(dirname(save_path))
    
    record["rendered_img"] = save_path
    save(save_path, rendered_img)
    return record
end

@gen function init_proposal(trace::Gen.DynamicDSLTrace, num_samples::Int)
    # Use importance sampling to initialize particle distribution
    scene_size, max_fireflies, steps = get_args(trace)
    _, observations = get_retval(trace)
    obs = get_choices(trace)[:observations => 1]
    chm = choicemap()
    chm[:observations=>1] = observations[1, :, :, :]
    observed_colors = StatsBase.mean(obs; dims=[2,3])[1, :, :]
    observed_colors = observed_colors ./ sum(observed_colors)
    
end

function smc(trace::Gen.DynamicDSLTrace, model::Gen.DynamicDSLFunction, num_particles::Int, num_samples::Int; record_json=true, experiment_tag="")
    scene_size, max_fireflies, steps = get_args(trace)
    gt_state, observations = get_retval(trace)

    obs = get_choices(trace)[:observations => 1]
    chm = choicemap()
    chm[:observations=>1] = observations[1, :, :, :]

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
            "gt_state" => gt_state,
            "smc_steps" => [[Dict() for _ in 1:num_particles] for _ in 1:steps]
        )

        particles = sample_unweighted_traces(state, 10)
        for i=1:10
            t = 1
            particle = particles[i]
            res_json["smc_steps"][t][i] = make_record(particle, savedir, t, i)
        end
    end

    mh_accepted = []
    println("Running SMC")
    println(repeat(".", steps-1), "| END")
    for t=2:steps
        # write out observation and save filepath name
        # record all the particle traces at time t - 1
    
        obs = get_choices(trace)[:observations => t]
        chm = choicemap()
        chm[:observations => t] = observations[t, :, :, :]

        resampled = Gen.maybe_resample!(state, ess_threshold=num_samples / 2)
        Gen.particle_filter_step!(state, (scene_size, max_fireflies, t,), (NoChange(), UnknownChange(), NoChange(),), chm)
        
        # mh_block_rejuvenation(state, t, obs)
        # data_driven_mcmc(state, obs, t, 10)
        mcmc_prior_rejuvenation(state, 1)

        if record_json
            particles = sample_unweighted_traces(state, 10)
            for i=1:10
                particle = particles[i]
                res_json["smc_steps"][t][i] = make_record(particle, savedir, t - 1, i)
            end
        end
    end

    if record_json
        save_path = joinpath(savedir, "results.json")
        println("http://localhost:8080/viz/viz.html?path=../$save_path")
        open(save_path, "w") do f
            JSON.print(f, res_json)
        end
    end

    # display(plot([mh_accepted], title="MH Acceptance Rate", xlabel="Step", ylabel="Acceptance Rate"))
    # return a sample of unweighted traces from the weighted collection
    return Gen.sample_unweighted_traces(state, num_samples)
end;

