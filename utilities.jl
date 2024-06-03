import Dates

function timestamp_dir(; base = "out/results", experiment_tag="")
    dir = nothing
    while isnothing(dir) || isdir(dir)
        date = Dates.format(Dates.now(), "yyyy-mm-dd")
        time = Dates.format(Dates.now(), "HH-MM-SS")
        experiment_tag = experiment_tag == "" ? "" : "_" * experiment_tag
        dir = joinpath(base, date, time * experiment_tag)
    end
    mkpath(dir)
    dir
end