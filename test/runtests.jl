using SinGAN
using Test

@testset "build_single_discrimiantor" begin
    noise_size = (64, 64, 3, 10)
    noise = randn(Float32, noise_size)
    dscr = SinGAN.build_single_discriminator(5, 128)
    @info dscr
    @test size(dscr(noise)) == (64, 64, 1, 10)
end

@testset "build_single_discrimiantor" begin
    noise_size = (64, 64, 3, 10)
    noise = randn(Float32, noise_size)
    gen = SinGAN.build_single_generator(5, 128)
    @info gen
    @test size(gen(noise)) == (64, 64, 3, 10)
end