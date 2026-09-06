# Taylor–Green vortex with Haftu refinement in the central square [0.25, 0.75]^2.
# The surrounding fluid returns gradually to the coarse resolution.
using TrixiParticles

particle_spacing = 0.02
refinement_region(x, t) = all(0.25 .<= x .<= 0.75)
particle_refinement = ParticleRefinementHaftu(refinement_region;
                                              particle_spacing=particle_spacing / 2)

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "taylor_green_vortex_2d.jl");
              wcsph=true, particle_spacing, particle_refinement,
              smoothing_length=1.2particle_spacing, perturb_coordinates=false)
