include("./distribution_utils.jl")
include("./firefly_models.jl")
include("./visualizations.jl")

function test_continuous_fireflies(scene_size, steps, max_fireflies)
    # Generate a trace from the model
    trace = simulate(continuous_fireflies_model, (scene_size, steps, max_fireflies))

    # Plot the trace
    anim = plot_multiple_fireflies(trace, firefly_size=6)
    gif(anim, "animations/continuous_fireflies.gif", fps=10)
end