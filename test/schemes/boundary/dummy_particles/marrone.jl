@testset verbose=true "`MarronePressureExtrapolation`" begin
    # Minimal setup with a single boundary particle, used by most testsets below.
    #
    # The boundary particle sits at `-e_1` and its normal is `-e_1/2`, so that the
    # interpolation point (the boundary particle mirrored across the wall) is
    # `coordinates - 2 * normals = e_1`.
    # A small cloud of fluid particles surrounds the interpolation point, so that the
    # MLS moment matrix at the interpolation point is regular. Fluid pressure, density
    # and velocity are prescribed by the individual testsets or by the linear fields
    # defined here, which the first-order MLS interpolation must reproduce exactly.
    #
    #                                        ○ ○ ○
    #        ● ------------ × ------------>  ○ ○ ○   fluid particles
    #      boundary   interpolation point    ○ ○ ○
    #      particle   (mirrored particle)
    #
    # The fluid has a gravity-like acceleration of `-2` in the first coordinate
    # direction, which activates the hydrostatic correction term of the method.
    function marrone_test_setup(D; viscosity=nothing, state_equation=nothing,
                                clip_negative_pressure=false, prescribed_motion=nothing)
        interpolation_point = zeros(D, 1)
        interpolation_point[1] = 1
        boundary_coordinates = -interpolation_point

        # Fluid particles on a 3^D grid centered around the interpolation point `e_1`,
        # all within the compact support of 2 * 0.4 = 0.8 around the interpolation point.
        fluid_coordinates = stack([collect(x)
                                   for x in Iterators.product(ntuple(i -> i == 1 ?
                                                                     [0.7, 1.1, 1.3] :
                                                                     [-0.2, 0.05, 0.3],
                                                                D)...)] |> vec)
        n_particles = size(fluid_coordinates, 2)

        # Velocity field that is linear in space, so that first-order MLS interpolation
        # must reproduce it exactly at the interpolation point.
        velocity = [dimension + 0.3 * fluid_coordinates[1, particle] -
                    0.2 * fluid_coordinates[D, particle]
                    for dimension in 1:D, particle in 1:n_particles]

        # Use varying masses to make sure that the volume weighting is tested as well.
        fluid = InitialCondition(; coordinates=fluid_coordinates, velocity,
                                 density=fill(1000.0, n_particles),
                                 mass=collect(range(0.5, 2.0; length=n_particles)),
                                 particle_spacing=0.1)
        smoothing_kernel = SchoenbergCubicSplineKernel{D}()
        fluid_system = WeaklyCompressibleSPHSystem(fluid; smoothing_kernel,
                                                   smoothing_length=0.4,
                                                   density_calculator=ContinuityDensity(),
                                                   state_equation,
                                                   acceleration=ntuple(i -> i == 1 ? -2.0 :
                                                                            0.0, D))

        boundary = InitialCondition(; coordinates=boundary_coordinates, density=1000.0,
                                    particle_spacing=0.1,
                                    normals=(boundary_coordinates - interpolation_point) /
                                            2)
        model = BoundaryModelDummyParticles(boundary.density, boundary.mass,
                                            MarronePressureExtrapolation(),
                                            smoothing_kernel, 0.4; viscosity,
                                            state_equation,
                                            clip_negative_pressure)
        wall = WallBoundarySystem(boundary, model; prescribed_motion)
        v_fluid = vcat(velocity, fluid.density')

        return (; fluid_system, wall, model, fluid_coordinates, boundary_coordinates,
                v_fluid, interpolation_point)
    end

    # The MLS interpolation is first-order consistent, so a pressure field that is linear
    # in space must be reproduced exactly at the interpolation point. The same holds for
    # the velocity field used for the no-slip wall velocity.
    @testset verbose=true "Linear Pressure and Velocity Fields" begin
        @testset "$D Dimensions" for D in (2, 3)
            viscosity = ViscosityAdami(nu=1.0e-6)
            setup = marrone_test_setup(D; viscosity)
            (; fluid_system, wall, model, fluid_coordinates, boundary_coordinates,
             v_fluid) = setup

            # Linear pressure field `p(x) = 6000 - 2000 x_1 + 3 x_D`.
            # At the interpolation point `e_1`, this yields `p = 6000 - 2000 = 4000`.
            fluid_system.pressure .= [6000 - 2000 * fluid_coordinates[1, particle] +
                                      3 * fluid_coordinates[D, particle]
                                      for particle in axes(fluid_coordinates, 2)]

            TrixiParticles.accumulate_marrone!(model, wall, fluid_system,
                                               boundary_coordinates, fluid_coordinates,
                                               v_fluid, DummySemidiscretization())
            TrixiParticles.finalize_marrone!(model, wall, zeros(1, 1), 1)

            # The hydrostatic correction transfers the pressure from the interpolation point
            # to the boundary particle:
            # `ρ a ⋅ (r_G - r_I) = 1000 * (-2, 0, …) ⋅ (-2, 0, …) = 4000`.
            # Note that this is consistent with evaluating the linear pressure field above
            # at the boundary particle `-e_1`, which yields `6000 + 2000 = 8000`.
            @test model.pressure[1] ≈ 8000

            # Without a state equation, the density is not updated (EDAC behavior).
            @test model.cache.density[1] == 1000

            # The velocity field `v_i(x) = i + 0.3 x_1 - 0.2 x_D` evaluates to `i + 0.3` at
            # the interpolation point `e_1`. The no-slip wall velocity of Adami et al. is
            # `2 v_wall - v_interpolated = -v_interpolated` for a wall at rest.
            @test isapprox(model.cache.wall_velocity[:, 1], -(collect(1:D) .+ 0.3))
        end
    end

    # When the fluid particles around an interpolation point do not span the full space
    # (e.g. when they are all aligned on a line or a plane, or close to a free surface),
    # the MLS moment matrix is singular and cannot be inverted. In this case, the
    # implementation falls back to zeroth-order (Shepard) interpolation, which still
    # reproduces constant fields exactly.
    @testset verbose=true "Fallback for Singular Moment Matrices" begin
        @testset "$D Dimensions" for D in (2, 3)
            N = D + 1

            # Moment matrix of a single (repeated) fluid neighbor with basis vector `basis`
            # and total weight 2. This matrix has rank 1 and is therefore singular.
            basis = SVector{N}(ntuple(i -> i == 1 ? 1.0 : 0.2, N))
            moment = 2 * basis * basis'
            coefficients = TrixiParticles.marrone_mls_coefficients(moment)

            # The right-hand side for the constant field `f ≡ 7` is `2 * basis * 7`.
            # Shepard interpolation must reproduce the constant exactly.
            @test isapprox(dot(coefficients, 2 * basis * 7), 7)

            # Without any fluid neighbors, the moment matrix is zero. The coefficients
            # must then be zero as well, so that the extrapolated pressure is zero
            # (and not `NaN`).
            @test iszero(TrixiParticles.marrone_mls_coefficients(zero(moment)))
        end
    end

    # Test the full update as it is called during a simulation, i.e. including the
    # inverse state equation, the optional pressure clipping and the reset of the
    # accumulators between time steps.
    @testset verbose=true "Full Update" begin
        @testset "`clip_negative_pressure=$clip_negative_pressure`" for clip_negative_pressure in
                                                                        (false, true)
            # Linear state equation, so that the inverse state equation is easy to verify:
            # `p = B (ρ/ρ₀ - 1)` with `B = ρ₀ c² = 1000 * 400 = 400_000`.
            state_equation = StateEquationCole(sound_speed=20.0, reference_density=1000.0,
                                               exponent=1, clip_negative_pressure=false)
            setup = marrone_test_setup(2; state_equation, clip_negative_pressure)
            (; fluid_system, wall, model) = setup
            semi = Semidiscretization(fluid_system, wall)
            ode = semidiscretize(semi, (0.0, 0.01))
            v_ode, u_ode = ode.u0.x
            v = TrixiParticles.wrap_v(v_ode, wall, semi)
            u = TrixiParticles.wrap_u(u_ode, wall, semi)

            # Constant negative fluid pressure. The hydrostatic correction adds +4000
            # (see above), so the boundary pressure is `-5000 + 4000 = -1000`.
            fluid_system.pressure .= -5000

            # Update twice to verify that the accumulators (moment matrix, right-hand sides)
            # are reset at the beginning of each update and results don't accumulate.
            for _ in 1:2
                TrixiParticles.update_pressure!(model, wall, v, u, v_ode, u_ode, semi)

                @test isapprox(model.pressure[1], (clip_negative_pressure ? 0 : -1000))

                # Inverse state equation: `ρ = ρ₀ (1 + p/B) = 1000 (1 - 1000/400_000) = 997.5`
                @test isapprox(model.cache.density[1],
                               (clip_negative_pressure ? 1000 : 997.5))
            end

            # Disable the interaction between the fluid system and the boundary system.
            # Without any fluid neighbors, the extrapolation must yield zero pressure
            # and the reference density.
            semi.interaction_matrix[2, 1] = false
            TrixiParticles.update_pressure!(model, wall, v, u, v_ode, u_ode, semi)

            @test model.pressure[1] == 0
            @test isapprox(model.cache.density[1], 1000)
        end
    end

    @testset verbose=true "Interpolation Points" begin
        # The interpolation points are computed from the normals as
        # `coordinates - 2 * normals`, which mirrors each boundary particle across the
        # wall surface.
        @testset "Mirrored Across the Wall Surface" begin
            setup = marrone_test_setup(2)
            @test setup.model.cache.interpolation_coordinates ≈ setup.interpolation_point
        end

        # When the wall moves, the interpolation points must follow the same motion,
        # so that they keep their position relative to the wall.
        @testset "Follow `PrescribedMotion`" begin
            # Rotation around the origin by the angle `t`.
            function rotation(x, t)
                return SVector(cos(t) * x[1] - sin(t) * x[2],
                               sin(t) * x[1] + cos(t) * x[2])
            end
            motion = PrescribedMotion(rotation, t -> true)
            setup = marrone_test_setup(2; prescribed_motion=motion)

            # Rotate by 90 degrees
            TrixiParticles.update_positions!(setup.wall, nothing, nothing, nothing,
                                             nothing, DummySemidiscretization(), pi / 2)

            # The boundary particle at `(-1, 0)` moves to `(0, -1)`
            @test isapprox(setup.wall.coordinates[:, 1], [0.0, -1.0], atol=1.0e-14)

            # The interpolation point at `(1, 0)` moves to `(0, 1)`
            @test isapprox(setup.model.cache.interpolation_coordinates[:, 1], [0.0, 1.0],
                           atol=1.0e-14)
        end
    end

    @testset verbose=true "Error Messages" begin
        # Every boundary particle requires a finite, nonzero normal to define its
        # interpolation point, so invalid normals must be rejected at construction time.
        @testset "Invalid Boundary Normals" begin
            kernel = SchoenbergCubicSplineKernel{2}()
            for normals in (nothing, zeros(2, 1), fill(NaN, 2, 1), fill(Inf, 2, 1))
                boundary = InitialCondition(; coordinates=[-0.1; 0.0;;], density=1000.0,
                                            particle_spacing=0.1, normals)
                model = BoundaryModelDummyParticles(boundary.density, boundary.mass,
                                                    MarronePressureExtrapolation(),
                                                    kernel, 0.1)
                @test_throws ArgumentError WallBoundarySystem(boundary, model)
            end
        end

        # The interpolation points are not particles of any system, so the neighborhood
        # search must support queries at arbitrary points. This rules out neighborhood
        # searches with precomputed neighbor lists.
        @testset "Unsupported Neighborhood Search" begin
            setup = marrone_test_setup(2)
            nhs = PrecomputedNeighborhoodSearch{2}()
            @test_throws ArgumentError Semidiscretization(setup.wall;
                                                          neighborhood_search=nhs)
        end

        # `RigidBodySystem` does not set up the interpolation points, so it must be
        # rejected rather than silently interpolating at the origin.
        @testset "Unsupported System Type" begin
            boundary = InitialCondition(; coordinates=[-0.1; 0.0;;], density=1000.0,
                                        particle_spacing=0.1, normals=[-0.05; 0.0;;])
            model = BoundaryModelDummyParticles(boundary.density, boundary.mass,
                                                MarronePressureExtrapolation(),
                                                SchoenbergCubicSplineKernel{2}(), 0.1)
            rigid_body = RigidBodySystem(boundary; boundary_model=model,
                                         particle_spacing=0.1)

            @test_throws ArgumentError Semidiscretization(rigid_body)
        end
    end

    @testset "show" begin
        setup = marrone_test_setup(2)
        @test repr(setup.model) ==
              "BoundaryModelDummyParticles(MarronePressureExtrapolation, Nothing)"
    end

    # For an elastic structure, the interpolation points are material points of the body,
    # so they have to follow its deformation. The offset `-2 * normal` is a material line
    # element and is therefore mapped to the current configuration by the deformation
    # gradient: `x_I = x_G - 2 F_G N_G`.
    @testset verbose=true "Elastic Structure" begin
        particle_spacing = 0.1
        smoothing_length = 2 * particle_spacing
        smoothing_kernel = SchoenbergCubicSplineKernel{2}()

        # A 3x3 elastic obstacle. `place_on_shell=true` is required by TLSPH, and
        # `compute_normals=true` provides the distance vectors from the surface of the
        # obstacle to each particle, pointing into the obstacle.
        solid = RectangularShape(particle_spacing, (3, 3), (0.0, 0.0); density=1000.0,
                                 place_on_shell=true, compute_normals=true)

        function tlsph_marrone_setup(; state_equation=nothing)
            model = BoundaryModelDummyParticles(solid.density, solid.mass,
                                                MarronePressureExtrapolation(),
                                                smoothing_kernel, smoothing_length;
                                                state_equation)
            system = TotalLagrangianSPHSystem(solid; smoothing_kernel, smoothing_length,
                                              young_modulus=1.0e6, poisson_ratio=0.3,
                                              boundary_model=model)

            return (; system, model)
        end

        # Prescribe a homogeneous deformation gradient and the corresponding
        # particle positions `x = F X`
        function apply_homogeneous_deformation!(system, deformation_grad)
            for particle in TrixiParticles.eachparticle(system)
                initial_position = TrixiParticles.extract_svector(solid.coordinates,
                                                                  system, particle)
                position = deformation_grad * initial_position

                for i in 1:2
                    system.current_coordinates[i, particle] = position[i]
                    for j in 1:2
                        system.deformation_grad[i, j, particle] = deformation_grad[i, j]
                    end
                end
            end
        end

        # Reference interpolation points, i.e. the boundary particles mirrored across the
        # surface of the undeformed obstacle
        reference_interpolation_coordinates = solid.coordinates .- 2 .* solid.normals

        # The TLSPH constructor must set up the interpolation points, just like the
        # `WallBoundarySystem` constructor does
        @testset "Initialization" begin
            (; model) = tlsph_marrone_setup()

            @test model.cache.normals == solid.normals
            @test model.cache.interpolation_coordinates ==
                  reference_interpolation_coordinates
        end

        # `F = I` must reproduce the undeformed mirror points
        @testset "Undeformed Body" begin
            (; system, model) = tlsph_marrone_setup()
            apply_homogeneous_deformation!(system, [1.0 0.0; 0.0 1.0])

            TrixiParticles.update_interpolation_coordinates!(model, system, nothing,
                                                             DummySemidiscretization())

            @test model.cache.interpolation_coordinates ≈
                  reference_interpolation_coordinates
        end

        # `F = R` must reproduce the rigid reflection, i.e. the same result as rotating the
        # undeformed mirror points. This is the compatibility check against the
        # `PrescribedMotion` path for rigid walls.
        @testset "Rigid Rotation" begin
            (; system, model) = tlsph_marrone_setup()
            angle = pi / 3
            rotation = [cos(angle) -sin(angle); sin(angle) cos(angle)]
            apply_homogeneous_deformation!(system, rotation)

            TrixiParticles.update_interpolation_coordinates!(model, system, nothing,
                                                             DummySemidiscretization())

            @test model.cache.interpolation_coordinates ≈
                  rotation * reference_interpolation_coordinates
        end

        # `F = λ I` scales the whole body, including the distance of each interpolation
        # point from the surface
        @testset "Uniform Stretch" begin
            (; system, model) = tlsph_marrone_setup()
            stretch = 1.7
            apply_homogeneous_deformation!(system, [stretch 0.0; 0.0 stretch])

            TrixiParticles.update_interpolation_coordinates!(model, system, nothing,
                                                             DummySemidiscretization())

            @test model.cache.interpolation_coordinates ≈
                  stretch * reference_interpolation_coordinates
        end

        # Simple shear tilts the offset, because `-2 F N` is the image of a material line
        # element and is no longer perpendicular to the deformed surface
        @testset "Simple Shear" begin
            (; system, model) = tlsph_marrone_setup()
            shear = [1.0 0.5; 0.0 1.0]
            apply_homogeneous_deformation!(system, shear)

            TrixiParticles.update_interpolation_coordinates!(model, system, nothing,
                                                             DummySemidiscretization())

            @test model.cache.interpolation_coordinates ≈
                  shear * reference_interpolation_coordinates
        end

        # Run the full update through a `Semidiscretization` with a surrounding fluid.
        # As for a wall, the combination of first-order MLS interpolation and hydrostatic
        # correction must reproduce a linear pressure field exactly at every structure
        # particle whose interpolation point is supported by the fluid, no matter how far
        # that point is from the particle.
        @testset "Pressure Extrapolation from a Surrounding Fluid" begin
            gravity = 9.81
            density = 1000.0
            state_equation = StateEquationCole(sound_speed=20.0,
                                               reference_density=density, exponent=1)
            hydrostatic_pressure(coords) = density * gravity * (1.0 - coords[2])

            # Fluid block with the obstacle cut out of it
            fluid_block = RectangularShape(particle_spacing, (16, 16), (-0.65, -0.65);
                                           density=density)
            fluid = setdiff(fluid_block, solid)

            (; system, model) = tlsph_marrone_setup(; state_equation)
            fluid_system = WeaklyCompressibleSPHSystem(fluid; smoothing_kernel,
                                                       smoothing_length,
                                                       density_calculator=ContinuityDensity(),
                                                       state_equation,
                                                       acceleration=(0.0, -gravity))

            semi = Semidiscretization(fluid_system, system)
            ode = semidiscretize(semi, (0.0, 0.01))
            v_ode, u_ode = ode.u0.x

            # `Semidiscretization` rebuilds the structure system to initialize its
            # self-interaction neighborhood search, so use the system it actually holds.
            # Both systems share the same boundary model object.
            system = semi.systems[2]
            v = TrixiParticles.wrap_v(v_ode, system, semi)
            u = TrixiParticles.wrap_u(u_ode, system, semi)

            fluid_system.pressure .= [hydrostatic_pressure(fluid.coordinates[:, particle])
                                      for particle in axes(fluid.coordinates, 2)]

            TrixiParticles.update_pressure!(model, system, v, u, v_ode, u_ode, semi)

            expected_pressure = [hydrostatic_pressure(solid.coordinates[:, particle])
                                 for particle in axes(solid.coordinates, 2)]

            # Particles whose interpolation point has no fluid neighbors fall back to zero
            # pressure, so only check the supported ones
            supported = [model.cache.moment_matrix[1, 1, particle] > eps()
                         for particle in TrixiParticles.eachparticle(system)]

            @test all(supported)
            @test all(isapprox.(model.pressure[supported],
                                expected_pressure[supported], atol=1.0e-9))

            # Evaluating the full right-hand side additionally verifies the update
            # ordering: the deformation gradient is computed in `update_quantities!`,
            # which runs before the boundary pressure is extrapolated.
            dv_ode = zero(v_ode)
            du_ode = zero(u_ode)
            TrixiParticles.kick!(dv_ode, v_ode, u_ode, ode.p, 0.0)
            TrixiParticles.drift!(du_ode, v_ode, u_ode, ode.p, 0.0)

            @test all(isfinite, dv_ode)
            @test all(isfinite, du_ode)
        end
    end

    # The testsets above verify the method on a single boundary particle with a
    # hand-crafted fluid cloud. Here, we verify it on a full rectangular tank, where
    # the boundary normals are computed by `RectangularTank` (including corners) and
    # where the interpolation points of the outer boundary layers have one-sided or
    # incomplete fluid support.
    @testset verbose=true "Rectangular Tank" begin
        particle_spacing = 1.0
        n_particles_xy = 10
        n_layers = 4
        width = particle_spacing * n_particles_xy
        height = particle_spacing * n_particles_xy
        density = 257.0
        gravity = 9.81

        smoothing_kernel = SchoenbergCubicSplineKernel{2}()
        smoothing_length = 3 * particle_spacing
        state_equation = StateEquationCole(sound_speed=10.0, reference_density=density,
                                           exponent=7)

        # Tank without a top face, so that the fluid has a free surface
        tank = RectangularTank(particle_spacing, (width, height), (width, height),
                               density, n_layers=n_layers,
                               faces=(true, true, true, false))

        # Set up the boundary and fluid systems and run one pressure extrapolation with
        # the fluid pressure and density prescribed by `pressure_function`.
        function extrapolate_pressure(pressure_function; acceleration=(0.0, 0.0))
            boundary_model = BoundaryModelDummyParticles(tank.boundary.density,
                                                         tank.boundary.mass,
                                                         MarronePressureExtrapolation(),
                                                         smoothing_kernel,
                                                         smoothing_length;
                                                         state_equation)
            boundary_system = WallBoundarySystem(tank.boundary, boundary_model)
            fluid_system = WeaklyCompressibleSPHSystem(tank.fluid; smoothing_kernel,
                                                       smoothing_length,
                                                       density_calculator=ContinuityDensity(),
                                                       state_equation, acceleration)

            semi = Semidiscretization(fluid_system, boundary_system)
            ode = semidiscretize(semi, (0.0, 0.01))
            v_ode, u_ode = ode.u0.x
            v = TrixiParticles.wrap_v(v_ode, boundary_system, semi)
            u = TrixiParticles.wrap_u(u_ode, boundary_system, semi)

            fluid_coordinates = tank.fluid.coordinates
            fluid_system.pressure .= [pressure_function(fluid_coordinates[:, particle])
                                      for particle in axes(fluid_coordinates, 2)]

            TrixiParticles.update_pressure!(boundary_model, boundary_system, v, u,
                                            v_ode, u_ode, semi)

            return boundary_model.pressure
        end

        # A constant pressure field must be reproduced exactly at every boundary
        # particle, including corners and outer layers with incomplete fluid support,
        # where the method falls back to Shepard interpolation.
        @testset "Constant Zero Pressure" begin
            pressure = extrapolate_pressure(coords -> 0.0)

            @test all(iszero, pressure)
        end

        @testset "Constant Non-Zero Pressure" begin
            pressure = extrapolate_pressure(coords -> 1234.5)

            @test all(isapprox.(pressure, 1234.5, atol=1.0e-9))
        end

        # The MLS interpolation is first-order consistent, so a linear (here:
        # hydrostatic) pressure field must be reproduced to machine precision.
        # Together with the hydrostatic correction term, the pressure at each boundary
        # particle must match the analytical pressure evaluated at that particle,
        # even though the interpolation is performed at the mirrored point.
        @testset "Hydrostatic Pressure Gradient" begin
            hydrostatic_pressure(coords) = density * gravity * (height - coords[2])

            pressure = extrapolate_pressure(hydrostatic_pressure;
                                            acceleration=(0.0, -gravity))

            boundary_coordinates = tank.boundary.coordinates
            expected_pressure = [hydrostatic_pressure(boundary_coordinates[:, particle])
                                 for particle in axes(boundary_coordinates, 2)]

            @test all(isapprox.(pressure, expected_pressure, atol=1.0e-8))
        end
    end
end
