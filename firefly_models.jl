using Gen
include("./distribution_utils.jl")

@gen function discrete_firefly_model(scene_size::Int, steps::Int)
    """
    This function describes a simple firefly model where a single firefly moves in a grid and blinks at a fixed frequency.

    Args:
    - scene_size: the size of the grid
    - steps: the number of steps to simulate

    Returns:
    - Dict with keys "xs", "ys", "blink_freq", "blinking"

    Traces over:
    - x: the x-coordinate of the firefly
    - y: the y-coordinate of the firefly
    - blinking: whether the firefly is blinking 
    """
    # sample number of fireflies
    xs = zeros(Int, steps)
    ys = zeros(Int, steps)
    vxs = zeros(Int, steps)
    vys = zeros(Int, steps)
    blinking = zeros(Int, steps)

    # Fix a blinking frequency
    blink_freq = 0.25

    for t in 1:steps
        # Update motion
        if t == 1
            # We'll initialize the position uniformly in our grid
            xs[t] = {(:x, t)} ~ uniform_discrete(0, scene_size)
            ys[t] = {(:y, t)} ~ uniform_discrete(0, scene_size)

        else
            prev_x = xs[t - 1]
            prev_y = ys[t - 1]
            
            xs[t] = {(:x, t)} ~ uniform_discrete(max(0, prev_x - 1), min(scene_size, prev_x + 1))
            ys[t] = {(:y, t)} ~ uniform_discrete(max(0, prev_y - 1), min(scene_size, prev_y + 1))
        end

        # Update blinking
        blinking[t] = {(:blinking, t)} ~ bernoulli(blink_freq)
    end
    
    return Dict("xs" => xs, "ys" => ys, "blink_freq" => blink_freq, "blinking" => blinking)
end


@gen function continuous_fireflies_model(scene_size::Int, steps::Int, max_fireflies::Int)
    """
    This function describes a simple firefly model where a single firefly moves in a random walk and blinks at a fixed frequency.

    Args:
    - scene_size: the size of the grid
    - steps: the number of steps to simulate
    - max_fireflies: Maximum number of fireflies

    Returns:
    - Dict with keys "xs", "ys", "blink_freq", "blinking"

    Traces over:
    - x: the x-coordinate of the firefly
    - y: the y-coordinate of the firefly
    - blinking: whether the firefly is blinking 
    """

    n_fireflies = {(:n_fireflies)} ~ uniform_discrete(1, max_fireflies)
        
    # sample number of fireflies
    xs = zeros(Float64, (n_fireflies, steps))
    ys = zeros(Float64, (n_fireflies, steps))
    vxs = zeros(Float64, (n_fireflies, steps))
    vys = zeros(Float64, (n_fireflies, steps))
    blinking = zeros(Int, (n_fireflies, steps))
    
    step_size = 1.0
    blink_freqs = zeros(Float64, n_fireflies)
    colors = zeros(Int, n_fireflies)

    blink_opts = [0.1, 0.15, 0.2, 0.25]
    
    # Initialize firefly colors and blinking frequencies
    for n in 1:n_fireflies
        blink_freq = {(:blink_freq, n)} ~ labeled_cat(blink_opts, uniprobs(blink_opts))
        blink_freqs[n] = blink_freq

        color = {(:color, n)} ~ uniform_discrete(1, max_fireflies)
        colors[n] = color
    end

    for t in 1:steps
        for n in 1:n_fireflies
            # Update motion
            if t == 1
                # We'll initialize the position uniformly in our grid
                xs[n, t] = {(:x, n, t)} ~ uniform(0, scene_size)
                ys[n, t] = {(:y, n, t)} ~ uniform(0, scene_size)
            else
                prev_x = xs[n, t - 1]
                prev_y = ys[n, t - 1]
                
                xs[n, t] = {(:x, n, t)} ~ truncated_normal(prev_x, step_size, 0, scene_size)
                ys[n, t] = {(:y, n, t)} ~ truncated_normal(prev_y, step_size, 0, scene_size)
            end

            # Update blinking
            blinking[n, t] = {(:blinking, n, t)} ~ bernoulli(blink_freqs[n])
        end
    end
    
    return Dict("xs" => xs, "ys" => ys, "blink_freqs" => blink_freqs, "blinking" => blinking, "colors" => colors)
end