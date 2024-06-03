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