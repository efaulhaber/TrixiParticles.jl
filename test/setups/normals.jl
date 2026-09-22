@testset "Compute Boundary Normal Vectors" begin
    @testset "2D `RectangularTank` Normals" begin
        particle_spacing = 1.0
        n_particles = 2
        n_layers = 1
        width = particle_spacing * n_particles
        height = particle_spacing * n_particles
        density = 257

        tank = RectangularTank(particle_spacing, (width, height), (width, height),
                               density, n_layers=n_layers,
                               faces=(true, true, true, false))

        (; normals) = tank.boundary
        normals_reference = [[-0.5 -0.5 0.5 0.5 0.0 0.0 -0.5 0.5]
                             [0.0 0.0 0.0 0.0 -0.5 -0.5 -0.5 -0.5]]

        @test normals == normals_reference
    end
    @testset "3D `RectangularTank` Normals" begin
        particle_spacing = 1.0
        n_particles = 2
        n_layers = 1
        tank_length = particle_spacing * n_particles
        density = 257

        tank = RectangularTank(particle_spacing,
                               (tank_length, tank_length, tank_length),
                               (tank_length, tank_length, tank_length),
                               density, n_layers=n_layers,
                               faces=(true, true, true, true, true, false))

        (; normals) = tank.boundary
        normals_reference = [[-0.5 -0.5 -0.5 -0.5 0.5 0.5 0.5 0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -0.5 -0.5 -0.5 -0.5 0.5 0.5 0.5 0.5 0.0 0.0 0.0 0.0 -0.5 -0.5 0.5 0.5 -0.5 -0.5 0.5 0.5]
                             [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -0.5 -0.5 -0.5 -0.5 0.5 0.5 0.5 0.5 0.0 0.0 0.0 0.0 -0.5 -0.5 0.5 0.5 -0.5 -0.5 0.5 0.5 -0.5 -0.5 0.5 0.5 0.0 0.0 0.0 0.0 -0.5 0.5 -0.5 0.5]
                             [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -0.5 -0.5 -0.5 -0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5]]
        @test normals == normals_reference
    end

    @testset "2D `SphereShape` Normals" begin
        particle_spacing = 0.5
        radius = 1.0
        center = (0.0, 0.0)
        density = 257

        sphere = SphereShape(particle_spacing, radius, center, density)
        (; normals) = sphere

        normals_reference = [[-0.207107 0.0 0.207107 -0.5 0.0 0.5 -0.207107 0.0 0.207107]
                             [-0.207107 -0.5 -0.207107 0.0 0.0 0.0 0.207107 0.5 0.207107]]

        @test all(isapprox.(sphere.normals, normals_reference, atol=1e-6))
    end
    @testset "3D `SphereShape` Normals" begin
        particle_spacing = 0.5
        radius = 1.0
        center = (0.0, 0.0, 0.0)
        density = 257

        sphere = SphereShape(particle_spacing, radius, center, density)

        (; normals) = sphere
        normals_reference = [[0.0 -0.207107 0.0 0.207107 0.0 -0.207107 0.0 0.207107 -0.5 0.0 0.5 -0.207107 0.0 0.207107 0.0 -0.207107 0.0 0.207107 0.0]
                             [-0.207107 0.0 0.0 0.0 0.207107 -0.207107 -0.5 -0.207107 0.0 0.0 0.0 0.207107 0.5 0.207107 -0.207107 0.0 0.0 0.0 0.207107]
                             [-0.207107 -0.207107 -0.5 -0.207107 -0.207107 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.207107 0.207107 0.5 0.207107 0.207107]]
        @test all(isapprox.(sphere.normals, normals_reference, atol=1e-6))
    end

    # The `RectangularShape` normals are the distance vectors from the surface of the
    # rectangle to each particle, where the surface lies half a particle spacing outside
    # of the outermost particles. Each normal points from the closest point on the surface
    # into the shape, which is the convention required by `MarronePressureExtrapolation`:
    # mirroring a particle as `coordinates - 2 * normals` moves it out of the shape.
    @testset verbose=true "`RectangularShape` Normals" begin
        # No normals are computed unless they are requested
        @testset "Disabled by Default" begin
            shape = RectangularShape(0.1, (3, 3), (0.0, 0.0), density=1000.0)

            @test isnothing(shape.normals)
        end

        @testset "2D" begin
            particle_spacing = 1.0

            # 4x3 particles at x = 0.5, 1.5, 2.5, 3.5 and y = 0.5, 1.5, 2.5,
            # so the surface is at x = 0, 4 and y = 0, 3
            shape = RectangularShape(particle_spacing, (4, 3), (0.0, 0.0), density=1000.0,
                                     compute_normals=true)

            # The two particles in the middle row are 1.5 away from the closest face in
            # both dimensions. Ties are resolved towards the first dimension.
            normals_reference = [[0.5 0.0 0.0 -0.5 0.5 1.5 -1.5 -0.5 0.5 0.0 0.0 -0.5]
                                 [0.0 0.5 0.5 0.0 0.0 0.0 0.0 0.0 0.0 -0.5 -0.5 0.0]]

            @test shape.normals == normals_reference

            # Mirroring across the surface must move every particle out of the shape
            mirrored = shape.coordinates .- 2 .* shape.normals
            outside = [!all(0 .<= mirrored[:, particle] .<= [4, 3])
                       for particle in axes(mirrored, 2)]

            @test all(outside)
        end

        @testset "3D" begin
            particle_spacing = 1.0

            # 3x3x3 particles, so the surface is at 0 and 3 in every dimension
            shape = RectangularShape(particle_spacing, (3, 3, 3), (0.0, 0.0, 0.0),
                                     density=1000.0, compute_normals=true)

            # Corner particle at (0.5, 0.5, 0.5): 0.5 from the closest face in every
            # dimension, so the tie is resolved towards the first dimension
            @test shape.coordinates[:, 1] == [0.5, 0.5, 0.5]
            @test shape.normals[:, 1] == [0.5, 0.0, 0.0]

            # Face particle at (1.5, 1.5, 0.5): closest to the face at z = 0
            @test shape.coordinates[:, 5] == [1.5, 1.5, 0.5]
            @test shape.normals[:, 5] == [0.0, 0.0, 0.5]

            # Center particle at (1.5, 1.5, 1.5): 1.5 from every face
            @test shape.coordinates[:, 14] == [1.5, 1.5, 1.5]
            @test shape.normals[:, 14] == [1.5, 0.0, 0.0]

            # Particle at (2.5, 1.5, 1.5): closest to the face at x = 3
            @test shape.coordinates[:, 15] == [2.5, 1.5, 1.5]
            @test shape.normals[:, 15] == [-0.5, 0.0, 0.0]

            # The normal length is the distance from the surface, so it is never smaller
            # than half a particle spacing
            @test all(>=(particle_spacing / 2),
                      [maximum(abs, shape.normals[:, particle])
                       for particle in axes(shape.normals, 2)])
        end

        # `place_on_shell=true` (required by `TotalLagrangianSPHSystem`) shifts the
        # particles by half a particle spacing, but the normals are computed from the
        # actual particle positions and are therefore unchanged.
        @testset "`place_on_shell=true`" begin
            particle_spacing = 1.0
            shape = RectangularShape(particle_spacing, (3, 3), (0.0, 0.0), density=1000.0,
                                     place_on_shell=true, compute_normals=true)
            shape_centered = RectangularShape(particle_spacing, (3, 3), (0.0, 0.0),
                                              density=1000.0, compute_normals=true)

            @test shape.coordinates == shape_centered.coordinates .- particle_spacing / 2
            @test shape.normals == shape_centered.normals
        end
    end
end
