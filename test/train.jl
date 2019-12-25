@testset "discriminator_loss" begin
    d_real = ones(Float32, 32, 32, 1, 1)
    d_g_fake = zeros(Float32, 32, 32, 1, 1)
    @test SinGAN.discriminator_loss(d_real, d_g_fake) ≈ 0f0
    @inferred Float32 SinGAN.discriminator_loss(d_real, d_g_fake)

    d_real = zeros(Float32, 32, 32, 1, 1)
    d_g_fake = ones(Float32, 32, 32, 1, 1)
    @test SinGAN.discriminator_loss(d_real, d_g_fake) ≈ 2f0
    @inferred Float32 SinGAN.discriminator_loss(d_real, d_g_fake)
    
    d_real = fill(0.5f0, 32, 32, 1, 1)
    d_g_fake = fill(0.5f0, 32, 32, 1, 1)
    @test SinGAN.discriminator_loss(d_real, d_g_fake) ≈ 0.5f0
    @inferred Float32 SinGAN.discriminator_loss(d_real, d_g_fake)
end

@testset "generator_adv_loss" begin
    d_g_fake = zeros(Float32, 32, 32, 1, 1)
    @test SinGAN.generator_adv_loss(d_g_fake) ≈ 1f0
    @inferred Float32 SinGAN.generator_adv_loss(d_g_fake)

    d_g_fake = ones(Float32, 32, 32, 1, 1)
    @test SinGAN.generator_adv_loss(d_g_fake) ≈ 0f0
    @inferred Float32 SinGAN.generator_adv_loss(d_g_fake)
end