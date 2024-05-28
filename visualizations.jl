using Gen
using Plots, Images

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

function plot_trace_and_observations(trace)
    args = get_args(trace)
    retvals = get_retval(trace)
    choices = get_choices(trace)
    max_fireflies = args[3]
    pixels = retvals["pixels"]
    n_fireflies = choices[:n_fireflies]
    xs = retvals["xs"]
    ys = retvals["ys"]
    scene_size = args[1]
    steps = args[2]

    fig = plot(layout=grid(1, 2), size=(800, 400), background_color=RGB(0, 0, 0), showaxis=true, ticks=false)
    gr()
    anim = Plots.@animate for step in 1:steps
        empty!(fig[1])
        empty!(fig[2])
        
        xlims!(0, scene_size)
        ylims!(0, scene_size)

        for n in 1:n_fireflies
            x = choices[(:x, n, step)]
            y = choices[(:y, n, step)]
            color = choices[(:color, n)]
            blinking = choices[(:blinking, n, step)]
            if blinking == 1
                scatter!(fig[1], [x], [y], color=color, markersize=4, label=nothing)
            else 
                scatter!(fig[1], [x], [y], color=color, markersize=4, label=nothing, markershape=:x)
            end
            if step > 1
                plot!(fig[1], xs[n, 1:step], ys[n, 1:step], color=color, label=nothing)
            end
        end
        
        function colormap(i)
            if i == 0
                return RGB(0, 0, 0)
            end
            color = palette(:default)[i]
        end

        # Julia's heatmap indexes the y-axis from 1 -> n (vs. plot which goes n -> 1)
        # So we need to flip the y-axis, and transpose pixels so they're (y, x) and not (x, y)
        frame = transpose(pixels[step, :, :] )[end:-1:1, :]
        frame = [colormap(pixelval) for pixelval in frame]
        # convert to RGB
        heatmap!(fig[2], frame, title="Observed Pixels (T=$step)")
        yflip!(fig[2])
    end
    
    return anim
end


function visualize_inference(gt_trace, inferred_traces; firefly_size=4)
    """
    Plots the trajectories of multiple fireflies in a grid.
    
    Args:
    - gt_trace: Actual trace of the firefly
    - inferred_traces: Inferred traces of the firefly
    - firefly_size: how large to make the firefly markers
    """

    args = get_args(gt_trace)
    retvals = get_retval(gt_trace)

    scene_size = args[1]
    steps = args[2]

    firefly_hues = retvals["colors"]
    xs = retvals["xs"]
    ys = retvals["ys"]
    blinks = retvals["blinking"]
    
    # Plot trajectories from inferred traces
    fig = plot(layout=grid(1, 2), size=(800, 400), background_color=RGB(0, 0, 0), showaxis=true, ticks=false)
    for trace in inferred_traces
        n_fireflies = get_choices(trace)[:n_fireflies]
        for t in 1:steps
            for n in 1:n_fireflies
                xs = get_retval(trace)["xs"]
                ys = get_retval(trace)["ys"]
                color = Int(get_retval(trace)["colors"][n])
                plot!(fig[1], xs[n, 1:t], ys[n, 1:t], color=color, markersize=firefly_size, label=nothing)
            end
        end
    end

end