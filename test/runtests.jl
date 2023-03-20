using Test
using Pixie
using LinearAlgebra
using Printf

include("test_util.jl")

const PIXIE_TEST = lowercase(get(ENV, "PIXIE_TEST", "all"))

@testset "All Tests" verbose=true begin
    if PIXIE_TEST in ("all", "unit", "unitandintegration")
        include("unit/unit_tests.jl")
    end

    if PIXIE_TEST in ("all", "integration", "unitandintegration")
        include("integration/integration_tests.jl")
    end

    if PIXIE_TEST in ("all", "system")
        include("system/system_tests.jl")
    end
end