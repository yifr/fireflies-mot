include("./distribution_utils.jl")
include("./firefly_models.jl")
include("./visualizations.jl")
include("./inference.jl")

function test_firefly_observation_model(scene_size, steps, max_fireflies)
    trace = simulate(firefly_gen_and_observe, (scene_size, steps, max_fireflies))
    anim = plot_trace_and_observations(trace)
    mp4(anim, "animations/firefly_observation.mp4", fps=10)
    
#     # # Plot the trace
#     # anim = plot_multiple_fireflies(trace, firefly_size=6)
#     # gif(anim, "animations/continuous_fireflies.gif", fps=10)

#     # anim = plot_multiple_fireflies(inferred_traces[1], firefly_size=6)
#     # gif(anim, "animations/inferred_continuous_fireflies.gif", fps=10)
end