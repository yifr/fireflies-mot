using Gen
using Hungarian
using StatsBase

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