@testset verbose=true "UpdateCallback" begin
    @testset verbose=true "show" begin
        # Default
        callback0 = UpdateCallback()

        show_compact = "UpdateCallback(interval=1)"
        @test repr(callback0) == show_compact

        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ UpdateCallback                                                                                   │
        │ ══════════════                                                                                   │
        │ interval: ……………………………………………………… 1                                                                │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", callback0) == show_box

        callback1 = UpdateCallback(interval=11)

        show_compact = "UpdateCallback(interval=11)"
        @test repr(callback1) == show_compact

        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ UpdateCallback                                                                                   │
        │ ══════════════                                                                                   │
        │ interval: ……………………………………………………… 11                                                               │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", callback1) == show_box

        callback2 = UpdateCallback(dt=1.2)

        show_compact = "UpdateCallback(dt=1.2)"
        @test repr(callback2) == show_compact

        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ UpdateCallback                                                                                   │
        │ ══════════════                                                                                   │
        │ dt: ……………………………………………………………………… 1.2                                                              │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", callback2) == show_box
    end

    @testset "Illegal Input" begin
        error_str = "Setting both interval and dt is not supported!"
        @test_throws ArgumentError(error_str) UpdateCallback(dt=0.1, interval=1)
    end

    @testset "Neighborhood Search Update Interval" begin
        semi_default = (; update_neighborhood_search_interval=0)
        semi_interval = (; update_neighborhood_search_interval=3)

        @test TrixiParticles.condition_update_nhs(semi_default,
                                                        (; stats=(; naccept=1)))
        @test TrixiParticles.condition_update_nhs(semi_interval,
                                                        (; stats=(; naccept=3)))
        @test !TrixiParticles.condition_update_nhs(semi_interval,
                                                         (; stats=(; naccept=4)))
    end
end
