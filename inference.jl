module FireflyInference
export smc

using Gen
using JSON
using Hungarian
include("./mcmc.jl")
include("./visualizations.jl")
include("./distribution_utils.jl")
include("./utilities.jl")

function normalize_weights(log_weights::Vector{Float64})
    log_total_weight = logsumexp(log_weights)
    log_normalized_weights = log_weights .- log_total_weight
    return (log_total_weight, log_normalized_weights)
end

function effective_sample_size(log_normalized_weights::Vector{Float64})
    log_ess = -logsumexp(2. * log_normalized_weights)
    return exp(log_ess)
end


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

function find_color_patches(image::Array{Float64,3}, threshold::Float64, size_prior::Int, luminance_threshold::Float64)
    height, width = size(image, 2), size(image, 3)
    visited = zeros(Int, height, width)
    patches = []

    # Function to flood fill and track patch
    function flood_fill(y, x)
        stack = [(y, x)]
        patch = []
        initial_color = image[:, y, x]
        too_large = false

        while !isempty(stack)
            cy, cx = pop!(stack)
            if cy < 1 || cy > height || cx < 1 || cx > width || visited[cy, cx] == 1
                continue
            end

            pixel_color = image[:, cy, cx]

            # Luminance filter and color filter
            if maximum(pixel_color) <= threshold || norm(pixel_color - initial_color) > luminance_threshold
                continue
            end

            visited[cy, cx] = 1
            push!(patch, (cx, cy, argmax(pixel_color)))

            for dy in -1:1, dx in -1:1
                if dy == 0 && dx == 0
                    continue
                end
                push!(stack, (cy + dy, cx + dx))
            end
        end

        if length(patch) > size_prior
            too_large = true
        end

        return patch, too_large
    end

    # Function to split large patches using simple median split
    function split_patch(patch)
        xs, ys = [p[1] for p in patch], [p[2] for p in patch]
        mean_x, mean_y = mean(xs), mean(ys)
        cluster1, cluster2 = [], []

        for p in patch
            if p[1] < mean_x || p[2] < mean_y
                push!(cluster1, p)
            else
                push!(cluster2, p)
            end
        end

        return cluster1, cluster2
    end

    # Main loop to find patches
    for y in 1:height, x in 1:width
        if maximum(image[:, y, x]) > threshold && visited[y, x] == 0
            patch, too_large = flood_fill(y, x)
            if too_large
                cluster1, cluster2 = split_patch(patch)
                push!(patches, cluster1, cluster2)
            else
                push!(patches, patch)
            end
        end
    end

    return patches, length(patches)
end

@gen function init_proposal(max_fireflies::Int, img_array::Array{Float64,3}, step::Int)
    #scene_size, max_fireflies, _ = get_args(prev_trace)
    scene_size = size(img_array)[2]
    
    # Sum across color channels and reshape to 2D
    # intensity = dropdims(sum(img_array, dims=1), dims=1)

    threshold = 0.2 # Luminance threshold
    size_prior = 30 # size prior for splitting patches
    luminance_threshold = 0.7 # color matching threshold
    patch_info, num_clusters = find_color_patches(img_array, threshold, size_prior, luminance_threshold)

    # For each cluster, find closest previous firefly of matching color
    # Update position accordingly
    for n in 1:num_clusters
        patch = patch_info[n]
        patch_index = uniform_discrete(1, length(patch))

        x_opts = ones(scene_size) .* 0.0001
        y_opts = ones(scene_size) .* 0.0001
        color_opts = ones(3) .* 0.0001
        color_opts[patch[patch_index][3]] = 1

        # Upweight x and y coordinates
        for p in patch
            x_coord = p[1]
            y_coord = p[2]
            x_opts[x_coord] += 1
            y_opts[y_coord] += 1
        end

        # Normalize
        x_opts = x_opts ./ sum(x_opts)
        y_opts = y_opts ./ sum(y_opts)
        color_opts = color_opts ./ sum(color_opts)

        x = {:states => step => :x => n} ~ categorical(x_opts)
        y = {:states => step => :y => n} ~ categorical(y_opts)

        init_x = {:init => :init_x => n} ~ categorical(x_opts)
        init_y = {:init => :init_y => n} ~ categorical(x_opts)
        color = {:init => :color => n} ~ categorical(color_opts)

        # Upweight blinking
        blinking = {:states => step => :blinking => n} ~ bernoulli(0.9)
    end
    
    # encode prior that there are at least as many fireflies as clusters
    # but possibly more
    n_firefly_probs = ones(max_fireflies) 
    n_firefly_probs[1:num_clusters] .= 0.1
    n_firefly_probs = n_firefly_probs ./ sum(n_firefly_probs)
    n_fireflies = {:init => :n_fireflies} ~ categorical(n_firefly_probs)

    # Any excess fireflies are probably not blinking
    for n in num_clusters + 1 : max_fireflies
        blinking = {:states => step => :blinking => n} ~ bernoulli(0.1)
    end
end

@gen function step_proposal(prev_trace, observation::Array{Float64,3}, step::Int)
    """
    Run detector on current frame
    Assign closest firefly to each detected patch with some probability
    """
    scene_size = 64
    threshold = 0.2 # Luminance threshold
    size_prior = 30 # size prior for splitting patches
    luminance_threshold = 0.7 # color matching threshold

    prev_choices = get_choices(prev_trace)
    n_fireflies = prev_choices[:init => :n_fireflies]

    patch_info, num_clusters = find_color_patches(observation, threshold, size_prior, luminance_threshold)

    # For each cluster, find closest previous firefly of matching color
    # Update position and velocity accordingly
    
    # Compute optimal assignment using hungarian algorithm
    cost_matrix = zeros(Float64, n_fireflies, num_clusters)
    patch_indices = zeros(Int, num_clusters)
    for k in 1:num_clusters
        patch_indices[k] = uniform_discrete(1, length(patch_info[k]))
    end

    for n in 1:n_fireflies
        for k in 1:num_clusters
            patch = patch_info[k][patch_indices[k]]
            # Compute cost as normalized distance between x,y, and color. 
            # If the distance is greater than the firefly could have moved in one or two steps, 
            # the cost is set to infinity
            prev_x = prev_choices[:states => step - 1 => :x => n]
            prev_y = prev_choices[:states => step - 1 => :y => n]
            l2_dist = norm([prev_x - patch[1], prev_y - patch[2]])

            vx_limit = (prev_choices[:states => step - 1 => :vx => n])
            vy_limit = (prev_choices[:states => step - 1 => :vy => n])
            if l2_dist > 2 * sqrt(vx_limit^2 + vy_limit^2)
                cost_matrix[n, k] = Inf
                continue
            end

            # Check color match
            color = prev_choices[:init => :color => n]
            if color != patch[3]
                cost_matrix[n, k] = Inf
                continue
            end

            cost_matrix[n, k] = l2_dist
        end
    end
    
    assignments, cost = Hungarian.hungarian(cost_matrix)
    for n in 1:n_fireflies
        k = assignments[n]
        if k != 0 && cost_matrix[n, k] != Inf # Check if the firefly is assigned to a cluster
            patch = patch_info[k][patch_indices[k]]
            obs_x = Float64(patch[1])
            obs_y = Float64(patch[2])
            x = {:states => step => :x => n} ~ trunc_norm(obs_x, 0.25, 1., Float64(scene_size))
            y = {:states => step => :y => n} ~ trunc_norm(obs_y, 0.25, 1., Float64(scene_size))

            prev_x = prev_choices[:states => step - 1 => :x => n]
            prev_y = prev_choices[:states => step - 1 => :y => n]
            vx = {:states => step => :vx => n} ~ trunc_norm(x - prev_x, 0.1, -3., 3.)
            vy = {:states => step => :vy => n} ~ trunc_norm(y - prev_y, 0.1, -3., 3.)
            blinking = {:states => step => :blinking => n} ~ bernoulli(0.9)
        else 
            blinking = {:states => step => :blinking => n} ~ bernoulli(0.01)
        end
    end
end 


function smc(trace::Gen.DynamicDSLTrace, model::Gen.DynamicDSLFunction, num_particles::Int, num_samples::Int; record_json=true, return_intermediate_traces=true, experiment_tag="")
    scene_size, max_fireflies, steps = get_args(trace)
    gt_state, observations = get_retval(trace)
    
    obs = get_choices(trace)[:observations=>1]
    chm = choicemap()
    chm[:observations=>1] = obs

    state = Gen.initialize_particle_filter(model, (scene_size, max_fireflies, 1), chm,
        init_proposal, (max_fireflies, obs, 1,), num_particles)

    intermediate_traces = []
    if return_intermediate_traces
        particles = copy(state.traces) #sample_unweighted_traces(state, num_samples)
        push!(intermediate_traces, particles)
    end
    
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
        for i = 1:10
            t = 1
            particle = particles[i]
            res_json["smc_steps"][t][i] = make_record(particle, savedir, t, i)
        end
    end

    mh_accepted = []
    println("Running SMC")
    println(repeat(".", steps - 1), "| END")
    ess_threshold = 0.5 * num_particles
    for t = 2:steps
        print(".")
        # write out observation and save filepath name
        # record all the particle traces at time t - 1

        obs = get_choices(trace)[:observations=>t]
        chm = choicemap()
        chm[:observations=>t] = obs

        argdiffs = (NoChange(), NoChange(), UnknownChange(),)
        Gen.particle_filter_step!(state, (scene_size, max_fireflies, t,), argdiffs, chm, step_proposal, (obs, t,))


        ess = effective_sample_size(normalize_weights(state.log_weights)[2])
        resampled = Gen.maybe_resample!(state, ess_threshold=ess_threshold)
        ess_threshold = ess * 0.5

        # mh_block_rejuvenation(state, t, obs)
        # data_driven_mcmc(state, obs, t, 10)
        # mcmc_prior_rejuvenation(state, 1)
        if return_intermediate_traces
            particles = copy(state.traces) #sample_unweighted_traces(state, num_samples)
            push!(intermediate_traces, particles)
        end

        if record_json
            particles = sample_unweighted_traces(state, 10)
            for i = 1:10
                particle = particles[i]
                res_json["smc_steps"][t][i] = make_record(particle, savedir, t - 1, i)
            end
        end
    end
    print("| END\n")
    if record_json
        save_path = joinpath(savedir, "results.json")
        println("http://localhost:8080/viz/viz.html?path=../$save_path")
        open(save_path, "w") do f
            JSON.print(f, res_json)
        end
    end

    if return_intermediate_traces
        # particles = state.traces #sample_unweighted_traces(state, num_samples)
        # push!(intermediate_traces, [particles])
        return intermediate_traces
    else
        return Gen.sample_unweighted_traces(state, num_samples)
    end
end;


end