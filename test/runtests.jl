using SinGAN

using Flux
using Test

# models

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

@testset "build_single_generator" begin
    noise_size = (64, 64, 3, 10)
    noise = randn(Float32, noise_size)
    gen = SinGAN.build_single_generator(5, 128)
    @info gen
    @test size(gen(noise)) == (64, 64, 3, 10)
end

@testset "Discriminator" begin
    @info SinGAN.Discriminator(4, 5)
end

@testset "NoiseConnection" begin
    @info SinGAN.NoiseConnection(Dense(1, 1), [1])
end

@testset "Generator" begin
    pyramid = SinGAN.size_pyramid(4/3, (32, 32), (42, 42))
    gen = SinGAN.Generator(pyramid, 5)
    @info gen
    noise_size = (32, 32, 3, 1)
    noise = randn(Float32, noise_size)
    @test typeof(gen(noise, 0, false)) <: AbstractArray{Float32, 4}
    @test size(gen(noise, 0, false)) == (32, 32, 3, 1)
    @test size(gen(noise, 0, true)) == (32, 32, 3, 1)
    @test size(gen(noise, 1, false)) == (32, 32, 3, 1)
    @test size(gen(noise, 1, true)) == (42, 42, 3, 1)
    @test size(gen(noise, 2, false)) == (42, 42, 3, 1)
    @test size(gen(noise, 2, true)) == (42, 42, 3, 1)

    adv_noise = [randn(Float32, 42, 42, 3, 1)]
    @test typeof(gen(noise, adv_noise, 2, false)) <: AbstractArray{Float32, 4}
    @test size(gen(noise, [], 1, false)) == (32, 32, 3, 1)
    @test size(gen(noise, [], 1, true)) == (42, 42, 3, 1)
    @test size(gen(noise, adv_noise, 2, false)) == (42, 42, 3, 1)
    @test size(gen(noise, adv_noise, 2, true)) == (42, 42, 3, 1)

end

# utils
@testset "zoom_image" begin
    x = randn(Float32, 2, 2, 3, 1)
    @info x
    @info SinGAN.zoom_image(x, (3, 3))
end