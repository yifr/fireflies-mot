using Gen
using Plots

function plot_multiple_fireflies(trace; plot_ground_truth=true, firefly_size=4)
    """
    Plots the trajectories of multiple fireflies in a grid.
    
    Args:
    - trace: a trace of the firefly model
    - plot_ground_truth: whether to plot the ground truth trajectory in a separate panel
    """

    args = get_args(trace)
    retvals = get_retval(trace)

    scene_size = args[1]
    steps = args[2]

    firefly_hues = retvals["colors"]
    xs = retvals["xs"]
    ys = retvals["ys"]
    blinks = retvals["blinking"]
    
    n_fireflies = get_choices(trace)[:n_fireflies]
    println("Plotting $n_fireflies fireflies"
            * " in a $scene_size x $scene_size grid"
            * " over $steps steps")

    if plot_ground_truth
        fig = plot(layout=grid(1, 2), size=(800, 400), background_color=RGB(0, 0, 0), showaxis=true, ticks=false)
        gr()
    else
        fig = plot(layout=grid(1, 1), size=(400, 400), background_color=RGB(0, 0, 0), showaxis=true, ticks=false)
    end

    xlims!(-1, scene_size + 1)
    ylims!(-1, scene_size + 1)

    anim = Plots.@animate for t in 1:steps
        if plot_ground_truth
            empty!(fig[2])
        else
            empty!(fig[1])
        end
        for n in 1:n_fireflies
            println("t=$t, n=$n")
            color = Int(firefly_hues[n])
            blinking = Int(blinks[n, t])

            if plot_ground_truth == true
                # Plot trajectory
                plot!(fig[1], xs[n, 1:t], ys[n, 1:t], color=color, markersize=firefly_size, label=nothing, title="Actual Location (T=$t)")
                if t > 1
                    markershape = blinking == 1 ? :circle : :x
                    scatter!(fig[1], [xs[n, t]], [ys[n, t]], color=color, markersize=firefly_size, label=nothing, markershape=markershape) 
                end
            end
            
            pane_idx = plot_ground_truth ? 2 : 1
            if blinking == 1
                scatter!(fig[pane_idx], [xs[n, t]], [ys[n, t]], color=color, markersize=firefly_size, label=nothing, title="Observed Location (T=$t)") 
            else
                scatter!(fig[pane_idx], [], [], title="Observed Location (T=$t)", label=nothing)
            end
            
        end
    end
    return anim
end