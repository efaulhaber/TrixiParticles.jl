function requires_update_callback(system::WeaklyCompressibleSPHSystem, semi)
    return !isnothing(system.particle_refinement) ||
           requires_update_callback(shifting_technique(system))
end

function timestep_smoothing_length(system::WeaklyCompressibleSPHSystem)
    isnothing(system.particle_refinement) && return initial_smoothing_length(system)
    return minimum(p -> smoothing_length(system, p), each_active_particle(system);
                   init=initial_smoothing_length(system))
end

function sort_refinement!(system::WeaklyCompressibleSPHSystem, perm)
    isnothing(system.particle_refinement) && return system
    system.mass .= system.mass[perm]
    for field in (:smoothing_length, :refinement_spacing, :refinement_mass_max)
        values = getproperty(system.cache, field)
        values .= values[perm]
    end
    return system
end

function update_particle_refinement!(system::WeaklyCompressibleSPHSystem,
                                     v_ode, u_ode, semi, t, integrator)
    refinement = system.particle_refinement
    isnothing(refinement) && return system
    semi.parallelization_backend isa KernelAbstractions.GPU &&
        throw(ArgumentError("ParticleRefinementHaftu currently supports only CPU execution"))
    length(semi.systems) == 1 ||
        throw(ArgumentError("ParticleRefinementHaftu currently requires a single fluid system"))
    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)
    refine_particles!(system, refinement, v, u, u_ode, semi, t)
    # Refresh pressure, shifting caches, and all neighbor searches before any RHS or output.
    update_systems_and_nhs(v_ode, u_ode, semi, t)
    derivative_discontinuity!(integrator, true)
    return system
end

@inline function refinement_neighbors(f::F, system, u, semi) where {F}
    # Each pass reads an immutable snapshot of its inputs. The topology-changing
    # split/merge operations below run serially and rebuild the active index list.
    foreach_point_neighbor(f, system, system, u, u, semi;
                           points=each_active_particle(system),
                           parallelization_backend=SerialBackend())
end

function refinement_spacing!(system, refinement, v, u, semi, t)
    (; refinement_spacing, refinement_spacing_tmp, refinement_mass_max) = system.cache
    coarse = system.initial_condition.particle_spacing
    fine = refinement.particle_spacing
    ratio = refinement.spacing_ratio
    nhs = get_neighborhood_search(system, semi)
    for i in each_active_particle(system)
        x = PointNeighbors.periodic_coords(current_coords(u, system, i), nhs.periodic_box)
        if refinement.region(x, t)
            refinement_spacing[i] = fine
        end
    end
    # Algorithm 1: use the geometric mean in strongly nonuniform neighborhoods.
    for i in each_active_particle(system)
        init = (oftype(coarse, Inf), zero(coarse), zero(coarse), 0)
        smin, smax, logsum, count = PointNeighbors.mapreduce_neighbor(
                                                                      refinement_spacing_reduce,
                                                                      u, u, nhs, i; init
                                                                      ) do _, j, _, distance
            if distance <
               compact_support(system.smoothing_kernel, smoothing_length(system, i))
                spacing = refinement_spacing[j]
                return (spacing, spacing, log(spacing), 1)
            end
            return init
        end
        x = PointNeighbors.periodic_coords(current_coords(u, system, i), nhs.periodic_box)
        target = refinement.region(x, t) ? fine :
                 (smax / smin < ratio^3 ? ratio * smin : exp(logsum / count))
        refinement_spacing_tmp[i] = clamp(target, fine, coarse)
    end
    for i in each_active_particle(system)
        refinement_spacing[i] = refinement_spacing_tmp[i]
        refinement_mass_max[i] = 1.05 * current_density(v, system, i) *
                                 refinement_spacing[i]^2
    end
    return system
end

function split_particles!(system, v, u)
    (; refinement_mass_max, refinement_spacing, smoothing_length) = system.cache
    parents = [i
               for i in each_active_particle(system)
               if system.mass[i] > refinement_mass_max[i]]
    slots = findall(!, system.buffer.active_particle)
    required = 6 * length(parents)
    length(slots) >= required ||
        throw(ArgumentError("particle refinement needs $required free buffer particles, but only $(length(slots)) are available; increase buffer_size"))
    next = 1
    for i in parents
        x = current_coords(u, system, i)
        h = smoothing_length[i]
        mass = system.mass[i] / 7
        rho = current_density(v, system, i)
        pressure = system.pressure[i]
        for k in 1:6
            j = slots[next]
            next += 1
            angle = k * pi / 3
            u[:, j] .= x + 0.4h * SVector(cos(angle), sin(angle))
            v[:, j] .= v[:, i]
            set_particle_density!(v, system, j, rho)
            system.pressure[j] = pressure
            system.mass[j] = mass
            smoothing_length[j] = 0.9h
            refinement_spacing[j] = refinement_spacing[i]
            refinement_mass_max[j] = refinement_mass_max[i]
            system.buffer.active_particle[j] = true
        end
        system.mass[i] = mass
        smoothing_length[i] = 0.9h
    end
    update_system_buffer!(system.buffer)
    return system
end

function merge_particles!(system, v, u, semi)
    (; refinement_partner, refinement_distance, refinement_mass_max,
     refinement_displacement, refinement_spacing, smoothing_length) = system.cache
    fill!(refinement_partner, 0)
    fill!(refinement_distance, Inf)
    refinement_neighbors(system, u, semi) do i, j, difference, distance
        i == j && return
        system.mass[i] <= refinement_mass_max[i] || return
        distance < (smoothing_length[i] + smoothing_length[j]) / 2 || return
        # Follow Algorithm 3's min (the surrounding prose instead says max).
        system.mass[i] + system.mass[j] <
        min(refinement_mass_max[i], refinement_mass_max[j]) || return
        if distance < refinement_distance[i] ||
           (distance == refinement_distance[i] && j < refinement_partner[i])
            refinement_partner[i] = j
            refinement_distance[i] = distance
            refinement_displacement[:, i] .= difference
        end
    end
    for i in each_active_particle(system)
        j = refinement_partner[i]
        j > i && refinement_partner[j] == i || continue
        mi, mj = system.mass[i], system.mass[j]
        mass = mi + mj
        fraction = mj / mass
        distance = refinement_distance[i]
        # Use the periodic minimum-image displacement, not the raw coordinates.
        u[:, i] .-= fraction .* refinement_displacement[:, i]
        v[:, i] .= (mi .* v[:, i] .+ mj .* v[:, j]) ./ mass
        if system.density_calculator isa SummationDensity
            rho = current_density(v, system)
            rho[i] = (mi * rho[i] + mj * rho[j]) / mass
        end
        system.pressure[i] = (mi * system.pressure[i] + mj * system.pressure[j]) / mass
        W = system.smoothing_kernel
        denominator = mi * kernel(W, fraction * distance, smoothing_length[i]) +
                      mj * kernel(W, (1 - fraction) * distance, smoothing_length[j])
        smoothing_length[i] = min(initial_smoothing_length(system),
                                  sqrt(mass * kernel(W, zero(distance), one(distance)) /
                                       denominator))
        refinement_spacing[i] = (mi * refinement_spacing[i] + mj * refinement_spacing[j]) /
                                mass
        refinement_mass_max[i] = min(refinement_mass_max[i], refinement_mass_max[j])
        system.mass[i] = mass
        system.mass[j] = 0
        deactivate_particle!(system, j, v, u)
    end
    update_system_buffer!(system.buffer)
    return system
end

function shift_refined_particles!(system, v, u, semi)
    (; refinement_displacement, refinement_gradient, reference_density) = system.cache
    fill!(refinement_displacement, 0)
    fill!(refinement_gradient, 0)
    W = system.smoothing_kernel
    # Equations (15), (16), (36), (37). Gradients and displacements are evaluated
    # before any particle is moved or any property is overwritten.
    refinement_neighbors(system, u, semi) do i, j, difference, distance
        i == j && return
        hi = smoothing_length(system, i)
        hij = (hi + smoothing_length(system, j)) / 2
        gradient = kernel_grad(W, difference, distance, hij)
        reference = kernel(W, particle_spacing(system, i), 0.759298480738450hij)
        weight = 1 + 0.24 * (kernel(W, distance, hij) / reference)^4
        displacement = -hi^2 / 2 * system.mass[j] / reference_density * weight * gradient
        refinement_displacement[:, i] .+= displacement
        volume = system.mass[j] / current_density(v, system, j)
        for dim in 1:2
            for component in 1:2
                refinement_gradient[component, dim, i] += volume *
                                                          (v[component, j] -
                                                           v[component, i]) * gradient[dim]
            end
            refinement_gradient[3, dim, i] += volume *
                                              (current_density(v, system, j) -
                                               current_density(v, system, i)) *
                                              gradient[dim]
        end
    end
    for i in each_active_particle(system)
        delta = SVector{2}(view(refinement_displacement, :, i))
        limit = 0.25 * smoothing_length(system, i)
        if norm(delta) > limit
            delta *= limit / norm(delta)
        end
        u[:, i] .+= delta
        for component in 1:2
            v[component, i] += dot(SVector{2}(view(refinement_gradient, component, :, i)),
                                   delta)
        end
        rho = current_density(v, system, i) +
              dot(SVector{2}(view(refinement_gradient, 3, :, i)), delta)
        set_particle_density!(v, system, i, rho)
    end
    return system
end

function refinement_smoothing_length!(system, u, semi)
    (; refinement_h_tmp, reference_density, smoothing_length_factor) = system.cache
    nhs = get_neighborhood_search(system, semi)
    for i in each_active_particle(system)
        init = (zero(eltype(system)), 0)
        mass, count = PointNeighbors.mapreduce_neighbor(
                                                        refinement_mass_reduce, u, u, nhs,
                                                        i; init) do _, j, _, distance
            if distance <
               compact_support(system.smoothing_kernel, smoothing_length(system, i))
                return (system.mass[j], 1)
            end
            return init
        end
        refinement_h_tmp[i] = min(initial_smoothing_length(system),
                                  smoothing_length_factor *
                                  sqrt(mass / (reference_density * count)))
    end
    for i in each_active_particle(system)
        system.cache.smoothing_length[i] = refinement_h_tmp[i]
    end
    return system
end

function refine_particles!(system, refinement, v, u, u_ode, semi, t)
    refinement_spacing!(system, refinement, v, u, semi, t)
    split_particles!(system, v, u)
    update_nhs!(semi, u_ode)
    for _ in 1:3
        merge_particles!(system, v, u, semi)
        update_nhs!(semi, u_ode)
    end
    for _ in 1:3
        shift_refined_particles!(system, v, u, semi)
        update_nhs!(semi, u_ode)
    end
    refinement_smoothing_length!(system, u, semi)
    return system
end

function check_refinement_configuration(system::WeaklyCompressibleSPHSystem, systems, nhs)
    isnothing(system.particle_refinement) && return nothing
    length(systems) == 1 ||
        throw(ArgumentError("ParticleRefinementHaftu currently requires a single fluid system"))
    if !(nhs isa GridNeighborhoodSearch{<:Any, <:Union{SerialUpdate, ParallelUpdate}})
        throw(ArgumentError("particle refinement requires GridNeighborhoodSearch with SerialUpdate() or ParallelUpdate()"))
    end
    isnothing(system.shifting_technique) ||
        throw(ArgumentError("ParticleRefinementHaftu includes shifting; set shifting_technique=nothing"))
    return nothing
end

function check_refinement_restart(system::WeaklyCompressibleSPHSystem)
    isnothing(system.particle_refinement) ||
        throw(ArgumentError("restarting with particle refinement is not supported"))
    return nothing
end

function write_refinement_vtk!(vtk, system::WeaklyCompressibleSPHSystem)
    isnothing(system.particle_refinement) && return vtk
    vtk["mass"] = [system.mass[i] for i in eachparticle(system)]
    vtk["smoothing_length"] = [smoothing_length(system, i) for i in eachparticle(system)]
    vtk["target_particle_spacing"] = [system.cache.refinement_spacing[i]
                                      for i in eachparticle(system)]
    return vtk
end

@inline function refinement_spacing_reduce(a, b)
    return (min(a[1], b[1]), max(a[2], b[2]), a[3] + b[3], a[4] + b[4])
end

@inline refinement_mass_reduce(a, b) = (a[1] + b[1], a[2] + b[2])

@inline uses_particle_refinement(system::WeaklyCompressibleSPHSystem) = !isnothing(system.particle_refinement)
