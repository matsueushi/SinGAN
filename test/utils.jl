@testset "zoom_pad_image" begin
    x = randn(Float32, 2, 2, 3, 1)
    @info x
    @info SinGAN.zoom_pad_image(x, (3, 3), (5, 5))
end

@testset "build_zero_pyramid" begin
    xs = Float32[1 2; 3 4] |> gpu
    ts = [(1, 1), (2, 2)]
    @test SinGAN.build_zero_pyramid(xs, ts) == [zeros(Float32, 1, 1, 3, 1), zeros(Float32, 2, 2, 3, 1)] |> gpu
end

@testset "build_noise_vector" begin
    xs = Float32[1 2; 3 4] |> gpu
    ts = [(1, 1), (2, 2)]
    as = [1f0, 0.5f0]
    @info SinGAN.build_noise_vector(xs, ts, as)
    # @code_warntype SinGAN.build_noise_vector(xs, ts, as)
end

@testset "rec_vector_generator" begin
    xs = Float32[1 2; 3 4] |> gpu
    ts = [(1, 1), (2, 2)]
    a = 1f0
    @info SinGAN.build_rec_vector(xs, ts, a)
    # @code_warntype SinGAN.build_rec_vector(xs, ts, a)
end
