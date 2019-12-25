@testset "zoom_image" begin
    x = randn(Float32, 2, 2, 3, 1)
    @info x
    @info SinGAN.zoom_image(x, (3, 3))
end

@testset "similar_zero_pyramid" begin
    xs = Float32[1 2; 3 4] |> gpu
    ts = [(1, 1), (2, 2)]
    @test SinGAN.similar_zero_pyramid(xs, ts) == [zeros(Float32, 1, 1, 3, 1), zeros(Float32, 2, 2, 3, 1)] |> gpu
    # @code_warntype SinGAN.similar_zero_pyramid(xs, ts)
end

@testset "noise_vector_generation" begin
    xs = Float32[1 2; 3 4] |> gpu
    ts = [(1, 1), (2, 2)]
    as = [1f0, 0.5f0]
    @info SinGAN.noise_vector_generation(xs, ts, as)
    # @code_warntype SinGAN.noise_vector_generation(xs, ts, as)
end

@testset "rec_vector_generator" begin
    xs = Float32[1 2; 3 4] |> gpu
    ts = [(1, 1), (2, 2)]
    a = 1f0
    @info SinGAN.rec_vector_generation(xs, ts, a)
    # @code_warntype SinGAN.rec_vector_generation(xs, ts, a)
end
