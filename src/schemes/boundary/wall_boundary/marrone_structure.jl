# `MarronePressureExtrapolation` for elastic structures.
# This is in a separate file because it requires the `TotalLagrangianSPHSystem`,
# which is defined after the boundary schemes.

# For an elastic structure, the interpolation points are deformed along with the body.
# The offset `-2 * normal` is a material line element, so it is mapped to the current
# configuration by the deformation gradient:
#
#   x_I = x_G + F_G (X_I - X_G) = x_G - 2 F_G N_G
#
# This reduces to the rigid reflection `x_G - 2 R N_G` for a rigid rotation `F_G = R`.
# Note that `F` is only required to be up to date here, which is guaranteed because
# `update_quantities!` (which computes the deformation gradient) runs before
# `update_boundary_interpolation!` (which calls this function).
function update_interpolation_coordinates!(model::BoundaryModelDummyParticles{MarronePressureExtrapolation},
                                           system::TotalLagrangianSPHSystem, u, semi)
    (; interpolation_coordinates, normals) = model.cache

    system_coordinates = current_coordinates(u, system)

    @threaded semi for particle in eachparticle(system)
        position = extract_svector(system_coordinates, system, particle)
        normal = extract_svector(normals, system, particle)
        deformation_grad = deformation_gradient(system, particle)

        interpolation_position = position - 2 * deformation_grad * normal

        for dimension in eachindex(interpolation_position)
            @inbounds interpolation_coordinates[dimension,
                                                particle] = interpolation_position[dimension]
        end
    end

    return model
end
