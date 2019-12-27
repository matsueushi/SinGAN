@testset "size_pyramid" begin
    @test SinGAN.size_pyramid(4/3, (32, 32), (400, 300)) == 
        [(32, 32), (42, 42), (56, 56), (75, 75), (101, 101), 
         (134, 134), (179, 179), (239, 239), (319, 300), (400, 300)]
    @test SinGAN.size_pyramid(4/3, (32, 32), (128, 128)) == 
        [(32, 32), (42, 42), (56, 56), (75, 75), (101, 101), (128, 128)]
end

@testset "channel_pyramid" begin
    @test SinGAN.channel_pyramid(5) == [32, 32, 32, 64, 64]
end

@testset "build_single_discrimiantor" begin
    noise_size = (64, 64, 3, 10)
    noise = randn(Float32, noise_size)
    dscr = SinGAN.build_single_discriminator(5, 128)
    @info dscr
    @test size(dscr(noise)) == (64, 64, 1, 10)
end

@testset "build_single_gen_layers" begin
    noise_size = (64, 64, 3, 10)
    noise = randn(Float32, noise_size)
    gen = SinGAN.build_single_gen_layers(5, 128)
    @info gen
    @test size(gen(noise)) == (64, 64, 3, 10)
end

@testset "DiscriminatorPyramid" begin
    @info SinGAN.DiscriminatorPyramid(4, 5)
end

@testset "NoiseConnection" begin
    nc = SinGAN.NoiseConnection(Conv((3, 3), 3 => 3, pad = 1), 5)
    @info nc
    prev = randn(Float32, 42, 42, 3, 1)
    noise = randn(Float32, 42, 42, 3, 1)
    @test size(nc(prev, noise)) == (32, 32, 3, 1)
    # @code_warntype nc(prev, noise)
end

@testset "GeneratorPyramid" begin
    image_shapes = [(32, 32), (44, 44)]
    genp = SinGAN.GeneratorPyramid(image_shapes, 5)
    @info genp
    @test genp.noise_shapes == [(42, 42), (54, 54)]
    xs = [randn(Float32, 42, 42, 3, 1), randn(Float32, 54, 54, 3, 1)] |> gpu
    @test size(genp(xs, 0, false)) == (32, 32, 3, 1)
    @test size(genp(xs, 0, true)) == (42, 42, 3, 1)
    @test size(genp(xs, 1, false)) == (32, 32, 3, 1)
    @test size(genp(xs, 1, true)) == (54, 54, 3, 1)
    @test size(genp(xs, 2, false)) == (44, 44, 3, 1)
end
