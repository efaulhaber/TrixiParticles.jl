@testset verbose=true "Haftu particle refinement" begin
    using OrdinaryDiffEqLowStorageRK
    using LinearAlgebra: norm
    TP = TrixiParticles

    function refinement_setup(; region=(x, t) -> all(0.25 .<= x .<= 0.75),
                              buffer_size=864, enabled=true,
                              density_calculator=ContinuityDensity(), n=12,
                              velocity=(x -> SVector(0.3, -0.2)))
        spacing = 1 / n
        ic = RectangularShape(spacing, (n, n), (0.0, 0.0);
                              density=1.0, velocity)
        refinement = enabled ?
                     ParticleRefinementHaftu(region; particle_spacing=spacing / 2) : nothing
        system = WeaklyCompressibleSPHSystem(ic; smoothing_length=1.2spacing,
                                             smoothing_kernel=SchoenbergQuinticSplineKernel{2}(),
                                             state_equation=StateEquationCole(sound_speed=10.0,
                                                                              reference_density=1.0,
                                                                              exponent=1),
                                             density_calculator,
                                             particle_refinement=refinement,
                                             buffer_size=enabled ? buffer_size : nothing,
                                             viscosity=ViscosityAdami(nu=0.01))
        periodic_box = PeriodicBox(min_corner=[0.0, 0.0], max_corner=[1.0, 1.0])
        semi = Semidiscretization(system;
                                  neighborhood_search=GridNeighborhoodSearch{2}(;
                                                                                periodic_box,
                                                                                update_strategy=SerialUpdate()),
                                  parallelization_backend=SerialBackend())
        ode = semidiscretize(semi, (0.0, 0.02))
        v, u = TP.wrap_v(ode.u0.x[1], system, semi), TP.wrap_u(ode.u0.x[2], system, semi)
        TP.update_systems_and_nhs(ode.u0.x..., semi, 0.0)
        return system, semi, ode, v, u
    end

    momentum(system, v) = sum(system.mass[i] * TP.current_velocity(v, system, i)
                              for i in TP.each_active_particle(system))
    mass(system) = sum(system.mass[i] for i in TP.each_active_particle(system))

    @testset "Validation and opt-out" begin
        @test_throws ArgumentError ParticleRefinementHaftu((x, t) -> true;
                                                           particle_spacing=0)
        @test_throws ArgumentError ParticleRefinementHaftu((x, t) -> true;
                                                           particle_spacing=0.1,
                                                           spacing_ratio=1)
        system, semi, ode, v, u = refinement_setup(enabled=false)
        @test isnothing(system.particle_refinement)
        @test isnothing(system.buffer)
        @test !TP.requires_update_callback(system, semi)
        before = deepcopy(ode.u0)
        @test TP.update_particle_refinement!(system, ode.u0.x..., semi, 0.0, nothing) ===
              system
        @test ode.u0 == before
        @test TP.smoothing_length(system, 1) == TP.initial_smoothing_length(system)
    end

    @testset "Split and merge conservation" begin
        for density_calculator in (ContinuityDensity(), SummationDensity())
            system, semi, ode, v, u = refinement_setup(; density_calculator)
            m0, p0 = mass(system), momentum(system, v)
            TP.refinement_spacing!(system, system.particle_refinement, v, u, semi, 0.0)
            count0 = length(TP.each_active_particle(system))
            TP.split_particles!(system, v, u)
            @test length(TP.each_active_particle(system)) > count0
            @test mass(system) ≈ m0
            @test momentum(system, v) ≈ p0
            TP.update_nhs!(semi, ode.u0.x[2])
            count_split = length(TP.each_active_particle(system))
            for _ in 1:3
                TP.merge_particles!(system, v, u, semi)
                TP.update_nhs!(semi, ode.u0.x[2])
            end
            @test length(TP.each_active_particle(system)) < count_split
            @test mass(system) ≈ m0
            @test momentum(system, v) ≈ p0
            TP.shift_refined_particles!(system, v, u, semi)
            @test momentum(system, v) ≈ p0
            @test all(i -> TP.current_velocity(v, system, i) ≈ SVector(0.3, -0.2),
                      TP.each_active_particle(system))
        end
    end

    @testset "Buffer exhaustion is atomic" begin
        system, semi, ode, v, u = refinement_setup(buffer_size=0)
        TP.refinement_spacing!(system, system.particle_refinement, v, u, semi, 0.0)
        before = deepcopy(ode.u0)
        masses = copy(system.mass)
        @test_throws ArgumentError TP.split_particles!(system, v, u)
        @test ode.u0 == before
        @test system.mass == masses
        @test length(TP.each_active_particle(system)) == 144
    end

    @testset "Variable kernel and sorting" begin
        system, semi, ode, v, u = refinement_setup()
        TP.refine_particles!(system, system.particle_refinement, v, u, ode.u0.x[2], semi,
                             0.0)
        @test minimum(system.cache.smoothing_length) < TP.initial_smoothing_length(system)
        @test TP.timestep_smoothing_length(system) < TP.initial_smoothing_length(system)
        i, j = first(TP.each_active_particle(system)), last(TP.each_active_particle(system))
        difference = SVector(0.1, 0.0)
        ga = TP.wcsph_kernel_grad(system, system, difference, norm(difference), i, j,
                                  system.particle_refinement)
        gb = TP.wcsph_kernel_grad(system, system, -difference, norm(difference), j, i,
                                  system.particle_refinement)
        @test ga == -gb
        @test TP.wcsph_kernel_grad(system, system, SVector(10.0, 0.0), 10.0, i, j,
                                   system.particle_refinement) == zero(difference)
        indices = collect(TP.each_active_particle(system))
        expected = [(system.mass[i], TP.smoothing_length(system, i),
                     TP.current_coords(u, system, i)) for i in reverse(indices)]
        TP.sort_system!(system, v, u, reverse(collect(eachindex(system.mass))),
                        system.buffer)
        actual = [(system.mass[i], TP.smoothing_length(system, i),
                   TP.current_coords(u, system, i))
                  for i in TP.each_active_particle(system)]
        @test actual == expected
        vtk = Dict{String, Any}()
        TP.write_refinement_vtk!(vtk, system)
        @test length(vtk["mass"]) == length(TP.each_active_particle(system))
        @test sum(vtk["mass"]) ≈ mass(system)
        @test vtk["smoothing_length"] ==
              [TP.smoothing_length(system, i) for i in TP.each_active_particle(system)]
        @test TP.requires_update_callback(system, semi)
        step_callback = StepsizeCallback(cfl=0.25).condition
        @test step_callback(nothing, 0.0, (; p=(; semi), opts=(; adaptive=false)))
        @test !step_callback(nothing, 0.0, (; p=(; semi), opts=(; adaptive=true)))
        @test_throws ArgumentError TP.check_refinement_restart(system)
        @test_throws ArgumentError Semidiscretization(system;
                                                      neighborhood_search=GridNeighborhoodSearch{2}())
    end

    @testset "Periodic merge and buffer reuse" begin
        system, semi, ode, v, u = refinement_setup()
        for i in 3:144
            TP.deactivate_particle!(system, i, v, u)
        end
        TP.update_system_buffer!(system.buffer)
        u[:, 1] .= (0.001, 0.5)
        u[:, 2] .= (0.999, 0.5)
        system.mass[1:2] .= 0.001
        system.cache.smoothing_length[1:2] .= 0.02
        system.cache.refinement_mass_max[1:2] .= 0.003
        TP.update_nhs!(semi, ode.u0.x[2])
        TP.merge_particles!(system, v, u, semi)
        @test collect(TP.each_active_particle(system)) == [1]
        @test abs(u[1, 1]) < 1e-14
        @test u[2, 1] == 0.5
        @test system.mass[1] == 0.002
        expected_h = sqrt(0.002 * TP.kernel(system.smoothing_kernel, 0.0, 1.0) /
                          (0.002 * TP.kernel(system.smoothing_kernel, 0.001, 0.02)))
        @test TP.smoothing_length(system, 1) ≈ expected_h
        system.cache.refinement_mass_max[1] = 0.001
        center = TP.current_coords(u, system, 1)
        TP.split_particles!(system, v, u)
        @test collect(TP.each_active_particle(system)) == collect(1:7)
        @test all(i -> norm(TP.current_coords(u, system, i) - center) ≈ 0.4expected_h, 2:7)
        @test mass(system) ≈ 0.002
    end

    @testset "Region can turn off and coarsen" begin
        system, semi, ode, v, u = refinement_setup(region=(x, t) -> t < 1 &&
                                                                    all(0.25 .<= x .<= 0.75))
        TP.refine_particles!(system, system.particle_refinement, v, u, ode.u0.x[2], semi,
                             0.0)
        refined_count = length(TP.each_active_particle(system))
        for _ in 1:20
            TP.refine_particles!(system, system.particle_refinement, v, u, ode.u0.x[2],
                                 semi, 2.0)
        end
        @test length(TP.each_active_particle(system)) < refined_count
        @test mass(system) ≈ 1
    end

    @testset "Taylor–Green vortex with a central square" begin
        velocity(x) = SVector(-cos(2pi * x[1]) * sin(2pi * x[2]),
                              sin(2pi * x[1]) * cos(2pi * x[2]))
        system, semi, ode, v, u = refinement_setup(; velocity, n=16, buffer_size=1536)
        m0 = mass(system)
        sol = solve(ode, RDPK3SpFSAL49(); dt=0.0005, adaptive=false,
                    callback=UpdateCallback(), save_everystep=false)
        @test sol.retcode == TP.SciMLBase.ReturnCode.Success
        @test sol.t[end] == 0.02
        @test mass(system) ≈ m0
        @test length(TP.each_active_particle(system)) > 256
        v = TP.wrap_v(sol.u[end].x[1], system, semi)
        u = TP.wrap_u(sol.u[end].x[2], system, semi)
        @test all(isfinite, v)
        @test all(i -> TP.current_density(v, system, i) > 0,
                  TP.each_active_particle(system))
        inside = [system.mass[i]
                  for i in TP.each_active_particle(system)
                  if all(0.35 .<= TP.current_coords(u, system, i) .<= 0.65)]
        outside = [system.mass[i]
                   for i in TP.each_active_particle(system)
                   if any(x -> x < 0.1 || x > 0.9, TP.current_coords(u, system, i))]
        @test !isempty(inside) && !isempty(outside)
        @test sum(inside) / length(inside) < sum(outside) / length(outside)
        error = sqrt(sum(system.mass[i] *
                         norm(TP.current_velocity(v, system, i) -
                              exp(-8pi^2 * 0.01 * sol.t[end]) *
                              velocity(TP.current_coords(u, system, i)))^2
                         for i in TP.each_active_particle(system)) / m0)
        @test error < 0.15
        @info "Adaptive Taylor–Green" particles=length(TP.each_active_particle(system)) velocity_l2_error=error mass=mass(system)
    end
end
