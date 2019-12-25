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

@testset "update_discriminator!" begin
    dscr = SinGAN.build_single_discriminator(3, 32)
    real_img = ones(Float32, 2, 2, 3, 1)
    g_fake = randn(Float32, 2, 2, 3, 1)
    opt = ADAM(0.0005f0)
    @inferred Float32 SinGAN.update_discriminator!(opt, dscr, real_img, g_fake)
    # @code_warntype(SinGAN.update_discriminator!(opt, dscr, real_img, g_fake))
end

@testset "update_generator!" begin
    dscr = SinGAN.build_single_discriminator(3, 4)
    nc_gen = SinGAN.build_single_nc_generator(3, 4, 1)
    real_img = ones(Float32, 2, 2, 3, 1)
    prev = randn(Float32, 2, 2, 3, 1)
    noise = randn(Float32, 4, 4, 3, 1)
    opt = ADAM(0.0005f0)
    alpha = 10f0
    @inferred Float32 SinGAN.update_generator!(opt, nc_gen, dscr, real_img, prev, noise, alpha)
    @code_warntype SinGAN.update_generator!(opt, nc_gen, dscr, real_img, prev, noise, alpha)
end