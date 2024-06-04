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
        n_fireflies = get_choices(particle)[:init=>:n_fireflies]
        run_mh(particles, i, select(:init => :n_fireflies), 3)

        # vary color 
        for n in 1:n_fireflies
            selection = select()
            push!(selection, :init => :color => n)
            run_mh(particles, i, selection, 3)

        end
        
        # Vary blink rate
        for n in 1:n_fireflies
            selection = select()
            push!(selection, :init => :blink_rate => n)
            run_mh(particles, i, selection, 2)

        end
        
        # Vary locations
        for n in 1:n_fireflies
            selection = select()
            push!(selection, :init => :init_x => n)
            push!(selection, :init => :init_y => n)
            for prev_t in 1:t - 1
                push!(selection, :states => prev_t => :x => n)
                push!(selection, :states => prev_t => :y => n)
            end
            run_mh(particles, i, selection, 10)
        end

        # Vary blinking states
        for n in 1:n_fireflies
            for prev_t in 1:t - 1
                run_mh(particles, i, select(:states => prev_t => :blinking => n), 2)
            end
        end
    end 
end

function mcmc_prior_rejuvenation(particles, mcmc_steps)
    num_particles = length(particles.traces)
    for i in 1:num_particles
        particle = particles.traces[i]
        run_mh(particles, i, selectall(), mcmc_steps)
    end
end