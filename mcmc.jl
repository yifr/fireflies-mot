using Gen

function run_mh(particles, i, selection, steps)
    for _ in 1:steps
        particles.traces[i], accepted = mh(particles.traces[i], selection)
    end
end

function mcmc_moves(particles, t)
    # select variables to change: n_fireflies, colors, blink_rates, blinking_states
    num_particles = length(particles.traces)
    for i in 1:num_particles
        particle = particles.traces[i]
        choices = get_choices(particle)

        # Vary number of fireflies
        @assert has_value(choices, :init => :n_fireflies)
        n_fireflies = get_choices(particle)[:init=>:n_fireflies]
        run_mh(particles, i, select(:init => :n_fireflies), 3)

        # vary color 
        selection = select()
        for n in 1:n_fireflies
            @assert has_value(choices, :init => :color => n)
            push!(selection, :init => :color => n)
        end
        run_mh(particles, i, selection, 3)
        
        # Vary blink rate
        selection = select()
        for n in 1:n_fireflies
            @assert has_value(choices, :init => :blink_rate => n)
            push!(selection, :init => :blink_rate => n)
        end
        run_mh(particles, i, selection, 2)

        # Vary starting position
        selection = select()
        for n in 1:n_fireflies
            @assert has_value(choices, :init => :init_x => n)
            push!(selection, :init => :init_x => n)
            push!(selection, :init => :init_y => n)
        end
        run_mh(particles, i, selection, 10)

        # Vary locations
        selection = select()
        for n in 1:n_fireflies
            for prev_t in 1:t - 1
                @assert has_value(choices, :states => prev_t => :x => n)
                push!(selection, :states => prev_t => :x => n)
                push!(selection, :states => prev_t => :y => n)
            end
        end
        run_mh(particles, i, selection, 10)

        # Vary blinking states
        selection = select()
        for n in 1:n_fireflies
            for prev_t in 1:t - 1
                @assert has_value(choices, :states => prev_t => :blinking => n)
                push!(selection, :states => prev_t => :blinking => n)
            end
        end
        run_mh(particles, i, selection, 20)
    end 
end

function mcmc_prior_rejuvenation(particles, mcmc_steps)
    num_particles = length(particles.traces)
    for i in 1:num_particles
        particle = particles.traces[i]
        run_mh(particles, i, selectall(), mcmc_steps)
    end
end