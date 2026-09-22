# The interpolation points are not particles of any system, so the fluid neighborhood
# search has to support queries at arbitrary points. This rules out neighborhood searches
# with precomputed neighbor lists.
function check_marrone_configuration(boundary_model, nhs)
    if boundary_model isa BoundaryModelDummyParticles{MarronePressureExtrapolation} &&
       first(PointNeighbors.requires_update(nhs))
        throw(ArgumentError("`MarronePressureExtrapolation` requires a neighborhood " *
                            "search supporting queries at arbitrary points, such as " *
                            "`GridNeighborhoodSearch` or `TrivialNeighborhoodSearch`."))
    end

    return nothing
end

function initialize_marrone!(model, initial_condition)
    return model
end

function initialize_marrone!(model::BoundaryModelDummyParticles{MarronePressureExtrapolation},
                             initial_condition)
    (; coordinates, normals) = initial_condition
    isnothing(normals) &&
        throw(ArgumentError("`MarronePressureExtrapolation` requires boundary normals"))

    (; interpolation_coordinates) = model.cache
    size(coordinates) == size(interpolation_coordinates) ||
        throw(ArgumentError("the boundary model and initial condition must have the same size"))
    all(isfinite, normals) || throw(ArgumentError("boundary normals must be finite"))

    for particle in axes(normals, 2)
        any(!iszero, view(normals, :, particle)) ||
            throw(ArgumentError("boundary normals must be nonzero for every particle"))
    end

    # Store the normals in the reference configuration. They are required in every update
    # to construct the interpolation points (see `update_interpolation_coordinates!`).
    model.cache.normals .= normals

    # Interpolation points in the reference configuration
    interpolation_coordinates .= coordinates .- 2 .* normals

    return model
end

# Reference position of the interpolation point of `particle`, i.e. the boundary particle
# mirrored across the wall surface in the reference configuration.
@propagate_inbounds function initial_interpolation_coordinates(model, system, particle)
    initial_position = extract_svector(initial_coordinates(system), system, particle)
    normal = extract_svector(model.cache.normals, system, particle)

    return initial_position - 2 * normal
end

# Update the points at which the fluid pressure is interpolated.
#
# For a wall at rest, the interpolation points never move, so this is a no-op and the
# points computed in `initialize_marrone!` are reused. Walls with a `PrescribedMotion`
# are handled in `update_marrone_interpolation_coordinates!`, which is called from
# `apply_prescribed_motion!` where the time `t` is available.
function update_interpolation_coordinates!(model, system, u, semi)
    return model
end

function update_marrone_interpolation_coordinates!(system, model, prescribed_motion, t,
                                                   semi)
    return model
end

function update_marrone_interpolation_coordinates!(system,
                                                   model::BoundaryModelDummyParticles{MarronePressureExtrapolation},
                                                   prescribed_motion::PrescribedMotion,
                                                   t, semi)
    system.ismoving[] || return model

    (; movement_function, moving_particles) = prescribed_motion
    (; interpolation_coordinates) = model.cache

    @threaded semi for particle in moving_particles
        initial_position = @inbounds initial_interpolation_coordinates(model, system,
                                                                       particle)
        position = movement_function(initial_position, t)
        for dimension in eachindex(position)
            @inbounds interpolation_coordinates[dimension, particle] = position[dimension]
        end
    end

    return model
end

function compute_pressure!(model, ::MarronePressureExtrapolation,
                           system, v, u, v_ode, u_ode, semi)
    (; cache, pressure) = model
    set_zero!(pressure)
    set_zero!(cache.moment_matrix)
    set_zero!(cache.pressure_rhs)
    set_zero!(cache.velocity_rhs)
    set_zero!(cache.volume)
    if haskey(cache, :wall_velocity)
        set_zero!(cache.wall_velocity)
    end

    # Move the interpolation points with the (possibly deforming) boundary
    update_interpolation_coordinates!(model, system, u, semi)

    system_coordinates = current_coordinates(u, system)

    @trixi_timeit timer() "compute boundary pressure" begin
        foreach_system(semi) do neighbor_system
            has_system_interaction(system, neighbor_system, semi) || return
            neighbor_system isa AbstractFluidSystem || return

            v_neighbor_system = wrap_v(v_ode, neighbor_system, semi)
            u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
            neighbor_coordinates = current_coordinates(u_neighbor_system, neighbor_system)

            accumulate_marrone!(model, system, neighbor_system, system_coordinates,
                                neighbor_coordinates, v_neighbor_system, semi)
        end
    end

    @trixi_timeit timer() "inverse state equation" @threaded semi for particle in
                                                                      eachparticle(system)
        finalize_marrone!(model, system, v, particle)
    end

    return model
end

function accumulate_marrone!(model, system, neighbor_system, system_coordinates,
                             neighbor_coordinates, v_neighbor_system, semi)
    interpolation_coordinates = model.cache.interpolation_coordinates

    foreach_point_neighbor(system, neighbor_system, interpolation_coordinates,
                           neighbor_coordinates, semi;
                           points=eachparticle(system)) do particle, neighbor,
                                                           pos_diff, distance
        @inbounds accumulate_marrone_pair!(model, system, neighbor_system,
                                           system_coordinates, v_neighbor_system,
                                           particle, neighbor, pos_diff, distance)
    end

    return model
end

@propagate_inbounds function accumulate_marrone_pair!(model, system, neighbor_system,
                                                      system_coordinates,
                                                      v_neighbor_system, particle,
                                                      neighbor, pos_diff, distance)
    (; cache, smoothing_length) = model
    NDIMS = ndims(system)

    # Scaling the linear basis by the smoothing length does not change the MLS
    # interpolant, but makes its conditioning independent of the unit of length.
    basis = SVector{NDIMS + 1}(ntuple(i -> i == 1 ? one(distance) :
                                           -pos_diff[i - 1] / smoothing_length,
                                      NDIMS + 1))
    density = current_density(v_neighbor_system, neighbor_system, neighbor)
    iszero(density) && return model

    volume = hydrodynamic_mass(neighbor_system, neighbor) / density
    weight = smoothing_kernel(model, distance, particle) * volume

    boundary_position = extract_svector(system_coordinates, system, particle)
    interpolation_position = extract_svector(cache.interpolation_coordinates, system,
                                             particle)
    acceleration = acceleration_source(neighbor_system) -
                   current_acceleration(system, particle)
    pressure = current_pressure(v_neighbor_system, neighbor_system, neighbor) +
               density * dot(acceleration, boundary_position - interpolation_position)
    velocity = current_velocity(v_neighbor_system, neighbor_system, neighbor)

    for i in 1:(NDIMS + 1)
        cache.pressure_rhs[i, particle] += weight * basis[i] * pressure
        for j in 1:(NDIMS + 1)
            cache.moment_matrix[i, j, particle] += weight * basis[i] * basis[j]
        end
        for dimension in 1:NDIMS
            velocity_contribution = weight * basis[i] * velocity[dimension]
            cache.velocity_rhs[i, dimension, particle] += velocity_contribution
        end
    end

    return model
end

# Return the coefficients that evaluate the MLS polynomial at the origin. For
# deficient support, use the constant part of the moment matrix (Shepard interpolation).
@inline function marrone_mls_coefficients(moment::SMatrix{N, N, ELTYPE}) where {N, ELTYPE}
    unit = SVector{N, ELTYPE}(ntuple(i -> i == 1 ? one(ELTYPE) : zero(ELTYPE), N))
    volume = moment[1, 1]
    volume > eps(ELTYPE) || return zero(unit)

    normalized_moment = moment / volume
    if abs(det(normalized_moment)) > sqrt(eps(ELTYPE))
        coefficients = inv(normalized_moment) * unit / volume
        all(isfinite, coefficients) && return coefficients
    end

    return unit / volume
end

function finalize_marrone!(model, system, v, particle)
    (; cache, pressure, viscosity, state_equation) = model
    NDIMS = ndims(system)
    N = NDIMS + 1
    ELTYPE = eltype(cache.density)

    moment = SMatrix{N, N, ELTYPE}(ntuple(i -> cache.moment_matrix[mod1(i, N), cld(i, N),
                                                                   particle], N * N))
    coefficients = marrone_mls_coefficients(moment)
    pressure_rhs = SVector{N, ELTYPE}(ntuple(i -> cache.pressure_rhs[i, particle], N))
    particle_pressure = dot(coefficients, pressure_rhs)

    if clip_negative_pressure(model)
        particle_pressure = max(particle_pressure, zero(particle_pressure))
    end
    pressure[particle] = particle_pressure
    inverse_state_equation!(cache.density, state_equation, pressure, particle)

    if !isnothing(viscosity) && !iszero(coefficients)
        velocity_rhs = SMatrix{N, NDIMS, ELTYPE}(ntuple(i -> cache.velocity_rhs[mod1(i, N),
                                                                                cld(i, N),
                                                                                particle],
                                                        N * NDIMS))
        interpolated_velocity = transpose(velocity_rhs) * coefficients
        for dimension in 1:NDIMS
            cache.wall_velocity[dimension, particle] = interpolated_velocity[dimension]
        end
        # `compute_wall_velocity!` expects the interpolated velocity in `wall_velocity`
        # and a normalization factor in `volume`. The MLS result is already normalized.
        cache.volume[particle] = one(ELTYPE)
        compute_wall_velocity!(viscosity, system, v, particle)
    end

    return model
end

# Shifting velocity of the boundary particles.
#
# Boundary particles are never shifted, but the shifting terms in the momentum and
# continuity equations require a shifting velocity for the boundary particles.
# Consistent with the Marrone boundary condition, the shifting velocity of the fluid is
# interpolated at the interpolation point (the boundary particle mirrored across the wall
# surface) and then mirrored across the wall surface as well, i.e., the normal component
# is flipped and the tangential components are kept.
# For a wall at `x = 0` with a fluid shifting velocity `(δu₁, δu₂)` at the interpolation
# point, this yields `(-δu₁, δu₂)` at the boundary particle.
@inline function delta_v_boundary(boundary_model, system, particle)
    return zero(SVector{ndims(system), eltype(system)})
end

@propagate_inbounds function delta_v_boundary(boundary_model::BoundaryModelDummyParticles{MarronePressureExtrapolation},
                                              system, particle)
    return extract_svector(boundary_model.cache.delta_v, system, particle)
end

@propagate_inbounds function delta_v(system::WallBoundarySystem, particle)
    return delta_v_boundary(system.boundary_model, system, particle)
end

function update_boundary_shifting!(system::WallBoundarySystem, v, u, v_ode, u_ode, semi, t)
    update_marrone_shifting!(system.boundary_model, system, v, u, v_ode, u_ode, semi)

    return system
end

function update_marrone_shifting!(model, system, v, u, v_ode, u_ode, semi)
    return model
end

function update_marrone_shifting!(model::BoundaryModelDummyParticles{MarronePressureExtrapolation},
                                  system, v, u, v_ode, u_ode, semi)
    (; cache) = model

    # Nothing to do when no fluid system uses a shifting technique
    has_shifting_neighbor(system, semi) || return model

    set_zero!(cache.delta_v)
    set_zero!(cache.delta_v_rhs)

    system_coordinates = current_coordinates(u, system)

    @trixi_timeit timer() "compute boundary shifting velocity" begin
        foreach_system(semi) do neighbor_system
            has_system_interaction(system, neighbor_system, semi) || return
            neighbor_system isa AbstractFluidSystem || return

            v_neighbor_system = wrap_v(v_ode, neighbor_system, semi)
            u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
            neighbor_coordinates = current_coordinates(u_neighbor_system, neighbor_system)

            accumulate_marrone_shifting!(model, system, neighbor_system,
                                         neighbor_coordinates, v_neighbor_system, semi)
        end

        @threaded semi for particle in eachparticle(system)
            @inbounds finalize_marrone_shifting!(model, system, system_coordinates,
                                                 particle)
        end
    end

    return model
end

# Whether any fluid system interacting with `system` uses a shifting technique.
# If not, the interpolated shifting velocity is zero anyway and the whole pass is skipped.
function has_shifting_neighbor(system, semi)
    result = Ref(false)

    foreach_system(semi) do neighbor_system
        has_system_interaction(system, neighbor_system, semi) || return
        neighbor_system isa AbstractFluidSystem || return

        if !isnothing(shifting_technique(neighbor_system))
            result[] = true
        end
    end

    return result[]
end

function accumulate_marrone_shifting!(model, system, neighbor_system,
                                      neighbor_coordinates, v_neighbor_system, semi)
    interpolation_coordinates = model.cache.interpolation_coordinates

    foreach_point_neighbor(system, neighbor_system, interpolation_coordinates,
                           neighbor_coordinates, semi;
                           points=eachparticle(system)) do particle, neighbor,
                                                           pos_diff, distance
        @inbounds accumulate_marrone_shifting_pair!(model, system, neighbor_system,
                                                    v_neighbor_system, particle, neighbor,
                                                    pos_diff, distance)
    end

    return model
end

@propagate_inbounds function accumulate_marrone_shifting_pair!(model, system,
                                                               neighbor_system,
                                                               v_neighbor_system, particle,
                                                               neighbor, pos_diff, distance)
    (; cache, smoothing_length) = model
    NDIMS = ndims(system)

    # Same basis and weights as in `accumulate_marrone_pair!`, so that the moment matrix
    # computed in `compute_pressure!` can be reused here.
    basis = SVector{NDIMS + 1}(ntuple(i -> i == 1 ? one(distance) :
                                           -pos_diff[i - 1] / smoothing_length,
                                      NDIMS + 1))
    density = current_density(v_neighbor_system, neighbor_system, neighbor)
    iszero(density) && return model

    volume = hydrodynamic_mass(neighbor_system, neighbor) / density
    weight = smoothing_kernel(model, distance, particle) * volume

    delta_v_neighbor = delta_v(neighbor_system, neighbor)

    for i in 1:(NDIMS + 1)
        for dimension in 1:NDIMS
            cache.delta_v_rhs[i, dimension, particle] += weight * basis[i] *
                                                         delta_v_neighbor[dimension]
        end
    end

    return model
end

@propagate_inbounds function finalize_marrone_shifting!(model, system, system_coordinates,
                                                        particle)
    (; cache) = model
    NDIMS = ndims(system)
    N = NDIMS + 1
    ELTYPE = eltype(cache.density)

    # The moment matrix is the same as for the pressure interpolation and has already
    # been computed in `compute_pressure!`.
    moment = SMatrix{N, N, ELTYPE}(ntuple(i -> cache.moment_matrix[mod1(i, N), cld(i, N),
                                                                   particle], N * N))
    coefficients = marrone_mls_coefficients(moment)

    delta_v_rhs = SMatrix{N, NDIMS, ELTYPE}(ntuple(i -> cache.delta_v_rhs[mod1(i, N),
                                                                          cld(i, N),
                                                                          particle],
                                                   N * NDIMS))
    interpolated_delta_v = transpose(delta_v_rhs) * coefficients

    # Unit normal of the wall surface in the current configuration.
    # The interpolation point is the boundary particle mirrored across the wall surface,
    # so the vector from the interpolation point to the boundary particle is normal
    # to the wall surface.
    boundary_position = extract_svector(system_coordinates, system, particle)
    interpolation_position = extract_svector(cache.interpolation_coordinates, system,
                                             particle)
    normal_direction = boundary_position - interpolation_position
    distance = norm(normal_direction)

    # Mirror the interpolated shifting velocity across the wall surface,
    # i.e., flip the normal component and keep the tangential components
    mirrored_delta_v = if distance > eps(ELTYPE)
        normal = normal_direction / distance
        interpolated_delta_v - 2 * dot(interpolated_delta_v, normal) * normal
    else
        interpolated_delta_v
    end

    for dimension in 1:NDIMS
        cache.delta_v[dimension, particle] = mirrored_delta_v[dimension]
    end

    return model
end
