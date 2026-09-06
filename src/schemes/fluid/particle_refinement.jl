"""
    ParticleRefinementHaftu(region; particle_spacing, spacing_ratio=1.1)

Adaptive splitting and merging following Algorithms 1–4 of Haftu, Muta and
Ramachandran, *Computer Physics Communications* 277 (2022), 108377,
https://doi.org/10.1016/j.cpc.2022.108377.

`region(x, t)` returns whether a particle belongs to the region with the finest
`particle_spacing`. Outside it, a graded transition to the initial particle spacing
is computed from neighboring particles, with adjacent spacing ratio `spacing_ratio`.
Coordinates passed to `region` are wrapped into the periodic box when present.

Pass this object as `particle_refinement` to a 2D [`WeaklyCompressibleSPHSystem`](@ref),
reserve daughter particles with `buffer_size`, and use [`UpdateCallback`](@ref).
Use `GridNeighborhoodSearch{2}(update_strategy=SerialUpdate())` (or `ParallelUpdate()`
with a compatible cell list), since incremental updates do not support inactive particles.
Place `UpdateCallback()` before output callbacks to save the adapted state.
Each update splits overweight particles into seven daughters, performs three mutual
nearest-neighbor merging passes and three shifting passes with Taylor corrections,
and updates individual smoothing lengths from neighboring masses. Splitting and
merging conserve mass and linear momentum; the Taylor correction during shifting
is not exactly momentum conserving.

Currently supports CPU execution, the quintic spline kernel, and a single fluid
system without boundaries, kernel corrections, or surface tension. This implements
the paper's adaptation procedure with the WCSPH equations (using a symmetric
variable-length kernel gradient), not its EDAC time evolution equations.

The initial particle spacing is the coarsest allowed spacing. The smoothing length
is bounded above by its initial value so the fixed neighbor search covers every
kernel. Buffer exhaustion raises an error before splitting any particles; increase
`buffer_size` in that case. Saved ODE states alone cannot restore adaptive masses
and smoothing lengths; adaptive restart is not supported.
"""
struct ParticleRefinementHaftu{F, T}
    region::F
    particle_spacing::T
    spacing_ratio::T
end

function ParticleRefinementHaftu(region; particle_spacing, spacing_ratio=1.1)
    spacing, ratio = promote(float(particle_spacing), float(spacing_ratio))
    isfinite(spacing) && spacing > 0 ||
        throw(ArgumentError("refinement particle_spacing must be finite and positive"))
    isfinite(ratio) && ratio > 1 ||
        throw(ArgumentError("spacing_ratio must be finite and greater than one"))
    return ParticleRefinementHaftu(region, spacing, ratio)
end

validate_refinement(::Nothing, args...) = nothing

function validate_refinement(refinement::ParticleRefinementHaftu, ic, kernel, h,
                             correction, surface_tension, surface_normal_method)
    ndims(ic) == 2 || throw(ArgumentError("ParticleRefinementHaftu supports only 2D"))
    kernel isa SchoenbergQuinticSplineKernel{2} ||
        throw(ArgumentError("ParticleRefinementHaftu requires SchoenbergQuinticSplineKernel{2}"))
    isfinite(h) && h > 0 || throw(ArgumentError("smoothing_length must be positive"))
    0 < refinement.particle_spacing < ic.particle_spacing ||
        throw(ArgumentError("refinement spacing must be smaller than the initial spacing"))
    all(isnothing, (correction, surface_tension, surface_normal_method)) ||
        throw(ArgumentError("particle refinement does not support corrections or surface tension"))
end

function create_cache_refinement(ic, refinement::ParticleRefinementHaftu, h)
    n = nparticles(ic)
    T = eltype(ic)
    return (; smoothing_length=fill(T(h), n),
            smoothing_length_factor=T(h / ic.particle_spacing),
            initial_smoothing_length=T(h), reference_density=first(ic.density),
            refinement_spacing=fill(T(ic.particle_spacing), n),
            refinement_spacing_tmp=zeros(T, n), refinement_mass_max=zeros(T, n),
            refinement_partner=zeros(Int, n), refinement_distance=fill(T(Inf), n),
            refinement_displacement=zeros(T, 2, n),
            refinement_gradient=zeros(T, 3, 2, n),
            refinement_h_tmp=zeros(T, n))
end

@inline smoothing_length(system, ::ParticleRefinementHaftu,
                         particle) = system.cache.smoothing_length[particle]
@inline initial_smoothing_length(system,
                                 ::ParticleRefinementHaftu) = system.cache.initial_smoothing_length

@inline timestep_smoothing_length(system) = initial_smoothing_length(system)

update_particle_refinement!(system, v_ode, u_ode, semi, t, integrator) = system
sort_refinement!(system, perm) = system
check_refinement_configuration(system, systems, nhs) = nothing
check_refinement_restart(system) = nothing
write_refinement_vtk!(vtk, system) = vtk
@inline uses_particle_refinement(system) = false

@inline function wcsph_kernel_grad(system, neighbor_system, pos_diff, distance,
                                   particle, neighbor, ::Nothing)
    return smoothing_kernel_grad_unsafe(system, pos_diff, distance, particle)
end

@inline function wcsph_kernel_grad(system, neighbor_system, pos_diff, distance,
                                   particle, neighbor, ::ParticleRefinementHaftu)
    # Averaging the two gradients preserves antisymmetry across resolution interfaces,
    # including when a pair is inside only one particle's support.
    return (smoothing_kernel_grad(system, pos_diff, distance, particle) -
            smoothing_kernel_grad(neighbor_system, -pos_diff, distance, neighbor)) / 2
end
