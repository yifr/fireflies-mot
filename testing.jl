include("./distribution_utils.jl")
include("./firefly_models.jl")
include("./visualizations.jl")
include("./inference.jl")

function test_firefly_observation_model(scene_size, steps, max_fireflies)
    trace = simulate(firefly_gen_and_observe, (scene_size, steps, max_fireflies))
    anim = plot_trace_and_observations(trace)
    mp4(anim, "animations/firefly_observation.mp4", fps=10)
    
    inferred_traces = particle_filter_default_proposal(trace, firefly_gen_and_observe, 1000, 10)
    visualize_inference(trace, inferred_traces)
    return inferred_traces
end