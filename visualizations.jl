using Gen
using Plots, Images
# include("./model.jl")
include("./utilities.jl")

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
            color = palette(:default)[Int(trunc(i))]
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

function visualize_inference(gt_trace, inferred_traces, steps; firefly_size=4)
    """
    Plots the trajectories of multiple fireflies in a grid.
    
    Args:
    - gt_trace: Actual trace of the firefly
    - inferred_traces: Inferred traces of the firefly
    - firefly_size: how large to make the firefly markers
    """

    args = get_args(gt_trace)
    retvals = get_retval(gt_trace)
    gt_choices = get_choices(gt_trace)
    scene_size = args[1]

    firefly_hues = retvals["colors"]
    gt_xs = retvals["xs"]
    gt_ys = retvals["ys"]
    blinks = retvals["blinking"]
    color_opts = [RGB(1, 0, 0), RGB(0, 1, 0), RGB(0, 0, 1)]
    # Plot trajectories from inferred traces
    fig = plot(layout=grid(1, 2), size=(800, 400), background_color=RGB(0, 0, 0), showaxis=true, ticks=false)
    anim = Plots.@animate for t in 1:steps 
        println("Plotting step $t")
        empty!(fig[1])
        empty!(fig[2])
        # plot ground truth location
        for n in 1:get_choices(gt_trace)[:n_fireflies]
            color_opt = Int(firefly_hues[n])
            color = color_opts[color_opt]
            blinking = Int(blinks[n, t])
            gt_x = gt_choices[(:x, n, t)]
            gt_y = gt_choices[(:y, n, t)]
            # plot!(fig[1], xs[n, 1:t], ys[n, 1:t], color=color, markersize=firefly_size, label=nothing)
            if blinking == 1
                scatter!(fig[1], [gt_x], [gt_y], color=color, markersize=firefly_size, markershape=:circle, label=nothing, xlims=(0, scene_size), ylims=(0, scene_size), yflip=true)
            else
                scatter!(fig[1], [gt_x], [gt_y], color=color, markersize=firefly_size, label=nothing, markershape=:x, xlims=(0, scene_size), ylims=(0, scene_size), yflip=true)
            end
        end

        for trace in inferred_traces
            n_fireflies = get_choices(trace)[:n_fireflies]
            for n in 1:n_fireflies
                xs = get_choices(trace)[(:x, n, t)]
                ys = get_choices(trace)[(:y, n, t)]
                color = Int(get_choices(trace)[(:color, n)])
                # plot!(fig[2], xs, ys color=color, markersize=firefly_size, label=nothing)
                blinking = get_choices(trace)[(:blinking, n, t)]
                if blinking == 1
                    scatter!(fig[2], [xs], [ys], color=color, markersize=firefly_size, label=nothing, xlims=(0, scene_size), ylims=(0, scene_size), yflip=true)
                else
                    scatter!(fig[2], [xs], [ys], color=color, markersize=firefly_size, label=nothing, markershape=:x, xlims=(0, scene_size), ylims=(0, scene_size), yflip=true)
                end
            end
        end
    end
    return anim

end

function animate_trace(trace)
    choices = get_choices(trace)
    scene_size, max_fireflies, steps = get_args(trace)
    frames = [mat_to_img(choices[:observations => t]) for t in range(1, steps)]
    y_size = size(frames[1])[1]
    x_size = size(frames[1])[2]
    fig = plot()
    n_fireflies = choices[:init => :n_fireflies]
    anim = Plots.@animate for t in range(1, steps)
        empty!(fig)
        frame = frames[t]
        heatmap!(fig, frame, xlims=(0, x_size + 1), ylims=(0, y_size + 1), yflip=true, background_color=:black, axis=false, grid=false)
        for n in 1:n_fireflies
            gt_x = choices[:states => t => :x => n]
            gt_y = choices[:states => t => :y => n]
            color_opt = choices[:init => :color => n]
            color = ["red", "green", "blue"][color_opt]
            blinking = choices[:states => t => :blinking => n]
            if blinking == 1
                scatter!(fig, [gt_x], [gt_y], color=color, markersize=4, markershape=:circle, label=nothing)
            else
                scatter!(fig, [gt_x], [gt_y], color=color, markersize=4, markershape=:x, label=nothing)
            end
        end
    end
    return anim
end

function animate_observations(frames; fps=10)
    n_frames, _, x_size, y_size = size(frames)
    if frames isa Array
        frames = [frames[i, :, :, :] for i in 1:n_frames]
    end
    
    fig = plot()
    anim = Plots.@animate for frame in frames
        empty!(fig)
        heatmap!(fig, frame, xlims=(0, x_size), ylims=(0, y_size), background_color=:black, axis=false, grid=false)
    end
    return anim
end

# function visualize_states(trace)
#     scene_size, max_fireflies, steps = get_args(trace)
#     choices = get_choices(trace)
#     states, observations = get_retval(trace)
#     n_fireflies = choices[:init => :n_fireflies]

function visualize_particles(particles, gt_trace)
    """
    animate trajectories for each particle. If there are 10 particles,
    and ground truth trace, animate 11 panes, with the first pane showing 
    the ground truth trace, and the remaining panes showing the particles.

    Ordering of the particles can be determined by the score of the particle.
    """

    scene_size, max_fireflies, steps = get_args(gt_trace)
    current_steps = get_args(particles[1])[3]
    gt_states, gt_observations = get_retval(gt_trace)
    gt_n_fireflies = get_choices(gt_trace)[:init=>:n_fireflies]
    num_particles = length(particles)
    
    fig = plot(layout=(2, floor(Int, num_particles//2) + 1), background_color=RGB(0, 0, 0), showaxis=true, ticks=false)
    gr()

    anim = @animate for t in 1:current_steps
        empty!(fig[1])
        xlims!(0, scene_size + 1)
        ylims!(0, scene_size + 1)
        # Plot ground truth states
        for n in 1:gt_n_fireflies
            x = gt_states[:xs][n, t]
            y = gt_states[:ys][n, t]
            color_opt = gt_states[:colors][n]
            color = ["red", "green", "blue"][color_opt]
            blinking = gt_states[:blinking_states][n, t]
            if blinking == 1
                scatter!(fig[1], [x], [y], color=color, markersize=4, label=nothing, title="Ground Truth", 
                titlefontsize=10, title_position=:center, 
                xlims=(0, scene_size), ylims=(0, scene_size), aspect_ratio=:equal, yflip=true)
            else 
                scatter!(fig[1], [x], [y], color=color, markersize=4, label=nothing, markershape=:x,
                xlims=(0, scene_size), ylims=(0, scene_size), aspect_ratio=:equal, yflip=true)
            end
            if t > 1
                # Draw trajectory after first time step
                plot!(fig[1], gt_states[:xs][n, 1:t], gt_states[:ys][n, 1:t], color=color, label=nothing, yflip=true)
            end
        end

        # Get states for each particle and plot
        states = [get_retval(particle)[1] for particle in particles]
        for p in 1:num_particles
            empty!(fig[2])
            xlims!(0, scene_size + 1)
            ylims!(0, scene_size + 1)
            n_fireflies = get_choices(particles[p])[:init=>:n_fireflies]

            for n in 1:n_fireflies
                x = states[p][:xs][n, t]
                y = states[p][:ys][n, t]
                color_opt = states[p][:colors][n]
                color = ["red", "green", "blue"][color_opt]
                blinking = states[p][:blinking_states][n, t]
                if blinking == 1
                    scatter!(fig[2], [x], [y], color=color, markersize=4, label=nothing, 
                    xlims=(0, scene_size), ylims=(0, scene_size), aspect_ratio=:equal, yflip=true)
                else 
                    scatter!(fig[2], [x], [y], color=color, markersize=4, label=nothing, markershape=:x, 
                    xlims=(0, scene_size), ylims=(0, scene_size), aspect_ratio=:equal, yflip=true)
                end
                if t > 1
                    plot!(fig[2], states[p][:xs][n, 1:t], states[p][:ys][n, 1:t], color=color, label=nothing, yflip=true)
                end
            end
        end
    end
    return anim
end


function visualize_particles_over_time(particles_over_time, gt_trace)
    """
    particles_over_time: t x n_particles vector
        - At each step we want to plot the final state of each particle, since that's 
          the prediction at that time step
    """

    scene_size, max_fireflies, steps = get_args(gt_trace)
    gt_states, gt_observations = get_retval(gt_trace)
    gt_n_fireflies = get_choices(gt_trace)[:init=>:n_fireflies]
    num_particles = length(particles_over_time[1])
    
    fig = plot(layout=(1, 2), background_color=RGB(0, 0, 0), showaxis=true, ticks=false)
    gr()
        
    anim = @animate for t in 1:steps
        empty!(fig[1])
        empty!(fig[2])

        xlims!(0, scene_size + 1)
        ylims!(0, scene_size + 1)
        # Plot ground truth states
        for n in 1:gt_n_fireflies
            x = gt_states[:xs][n, t]
            y = gt_states[:ys][n, t]
            color_opt = gt_states[:colors][n]
            color = ["red", "green", "blue"][color_opt]
            blinking = gt_states[:blinking_states][n, t]
            if blinking == 1
                scatter!(fig[1], [x], [y], color=color, markersize=4, label=nothing, title="GT", 
                titlefontsize=10, title_position=:center, 
                xlims=(0, scene_size), ylims=(0, scene_size), aspect_ratio=:equal, yflip=true)
            else 
                scatter!(fig[1], [x], [y], color=color, markersize=4, label=nothing, markershape=:x,
                xlims=(0, scene_size), ylims=(0, scene_size), aspect_ratio=:equal, yflip=true)
            end
            if t > 1
                # Draw trajectory after first time step
                plot!(fig[1], gt_states[:xs][n, 1:t], gt_states[:ys][n, 1:t], color=color, label=nothing, yflip=true)
            end
        end

        # Get states for each particle and plot
        particles = particles_over_time[t]
        n_particles = length(particles)
        states = [get_retval(particle)[1] for particle in particles]        

        scores = [get_score(particle) for particle in particles]
        scores = scores ./ sum(scores) # normalize
        scores = max.(scores, 0.1)

        xlims!(0, scene_size + 1)
        ylims!(0, scene_size + 1)

        for p in 1:num_particles
            particle = particles[p]
            choices = get_choices(particle)
            n_fireflies = choices[:init=>:n_fireflies]
            score = scores[p]
            for n in 1:n_fireflies
                x = states[p][:xs][n, t]
                y = states[p][:ys][n, t]
                color_opt = states[p][:colors][n]
                color = ["red", "green", "blue"][color_opt]
                blinking = choices[:states => t => :blinking => n]
                if isnan(score)
                    println("Score is NaN")
                    score = 1/num_particles
                end
                if blinking == 1
                    scatter!(fig[2], [x], [y], color=color, markersize=4, label=nothing, 
                    xlims=(0, scene_size), ylims=(0, scene_size), aspect_ratio=:equal, title=t, yflip=true,
                    alpha=score)
                else 
                    scatter!(fig[2], [x], [y], color=color, markersize=4, label=nothing, markershape=:x, 
                    xlims=(0, scene_size), ylims=(0, scene_size), aspect_ratio=:equal, title=t, yflip=true,
                    alpha=score)
                end
                if t > 1
                    plot!(fig[2], states[p][:xs][n, 1:t], states[p][:ys][n, 1:t], color=color, label=nothing, yflip=true,
                    alpha=score)
                end
            end
        end
    end
    return anim
end
