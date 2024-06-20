import Dates

function timestamp_dir(; base = "out/results", experiment_tag="")
    dir = nothing
    while isnothing(dir) || isdir(dir)
        date = Dates.format(Dates.now(), "yyyy-mm-dd")
        time = Dates.format(Dates.now(), "HH-MM-SS")
        dir = experiment_tag == "" ? joinpath(base, date, time) : joinpath(base, experiment_tag, date, time)
    end
    mkpath(dir)
    dir
end

function clip(x, low, high)
    return max(low, min(x, high))
end

function mat_to_img(mat)
    """
    Convert matrix to image
    """
    img = colorview(RGB, clip.(mat, 0., 1.))
    return img
end

function find_fireflies(image::Array{Float64, 3})
    """
    Extract locations and colors of fireflies in an image
    """

    # Convert image to grayscale and normalize
    image_luminance = Gray.(image)
    image_luminance = image_luminance ./ maximum(image_luminance)
    
    # Find peak luminance
    locs = findall(image_luminance .== 1)
    locs = [(loc[1], loc[2]) for loc in locs]
    
    # filter duplicate locations
    is_adjacent(loc1, loc2) = norm(loc1 .- loc2) <= sqrt(2)
    
    # Filter adjacent locations
    filtered_locs = []
    for loc in locs
        if !any(is_adjacent(loc, other_loc) for other_loc in filtered_locs)
            push!(filtered_locs, loc)
        end
    end

    colors = []
    for loc in filtered_locs
        x, y = loc[1], loc[2]
        color = image[:, y, x]
        push!(colors, color)
    end
    return filtered_locs, colors
end

