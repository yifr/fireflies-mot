include("./distribution_utils.jl")
include("./firefly_models.jl")
include("./visualizations.jl")
include("./inference.jl")

function test_firefly_observation_model(scene_size, steps, max_fireflies; experiment_tag="")
    trace = simulate(firefly_gen_and_observe, (scene_size, steps, max_fireflies))
    anim = plot_trace_and_observations(trace)
    mp4(anim, "animations/firefly_observation$experiment_tag.mp4", fps=10)
    
    inferred_traces = particle_filter_rejuv_resim(trace, firefly_gen_and_observe, 100, 10)
    score = 0
    for trace in inferred_traces
        score += get_score(trace)
    end
    println("Average score: ", score / length(inferred_traces))
    anim = visualize_inference(trace, inferred_traces, steps)
    mp4(anim, "animations/firefly_inference$experiment_tag.mp4", fps=10)
    return inferred_traces
end