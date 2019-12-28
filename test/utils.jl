@testset "size_pyramid" begin
    @test SinGAN.size_pyramid(4 / 3, (32, 32), (400, 300)) == 
        [(32, 32), (42, 42), (56, 56), (75, 75), (101, 101), 
         (134, 134), (179, 179), (239, 239), (319, 300), (400, 300)]
    @test SinGAN.size_pyramid(4 / 3, (32, 32), (128, 128)) == 
        [(32, 32), (42, 42), (56, 56), (75, 75), (101, 101), (128, 128)]
    @code_warntype SinGAN.size_pyramid(4 / 3, (32, 32), (128, 128))
end

@testset "channel_pyramid" begin
    @test SinGAN.channel_pyramid(5) == [32, 32, 32, 64, 64]
    @code_warntype SinGAN.channel_pyramid(5)
end

@testset "resize_and_padding" begin
    x = randn(Float32, 2, 2, 3, 1)
    @info x
    @info SinGAN.resize_and_padding(x, (3, 3), (5, 5))
    # @code_warntype SinGAN.resize_and_padding(x, (3, 3), (5, 5))
end

@testset "build_zero_pyramid" begin
    xs = Float32[1 2; 3 4] |> gpu
    ts = [(1, 1), (2, 2)]
    @test SinGAN.build_zero_pyramid(xs, ts) == [zeros(Float32, 1, 1, 3, 1), zeros(Float32, 2, 2, 3, 1)] |> gpu
    # @code_warntype SinGAN.build_zero_pyramid(xs, ts)
end

@testset "build_noise_pyramid" begin
    xs = Float32[1 2; 3 4] |> gpu
    ts = [(1, 1), (2, 2)]
    as = [1f0, 0.5f0]
    @info SinGAN.build_noise_pyramid(xs, ts, as)
    # @code_warntype SinGAN.build_noise_pyramid(xs, ts, as)
end

@testset "rec_vector_generator" begin
    xs = Float32[1 2; 3 4] |> gpu
    ts = [(1, 1), (2, 2)]
    a = 1f0
    @info SinGAN.build_rec_pyramid(xs, ts, a)
    # @code_warntype SinGAN.build_rec_pyramid(xs, ts, a)
end
