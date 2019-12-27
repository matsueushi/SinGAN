@testset "discriminator_loss" begin
    d_real = ones(Float32, 32, 32, 1, 1)
    d_g_fake_adv = zeros(Float32, 32, 32, 1, 1)
    @test SinGAN.discriminator_loss(d_real, d_g_fake_adv) ≈ 0f0
    @inferred Float32 SinGAN.discriminator_loss(d_real, d_g_fake_adv)

    d_real = zeros(Float32, 32, 32, 1, 1)
    d_g_fake_adv = ones(Float32, 32, 32, 1, 1)
    @test SinGAN.discriminator_loss(d_real, d_g_fake_adv) ≈ 2f0
    @inferred Float32 SinGAN.discriminator_loss(d_real, d_g_fake_adv)
    
    d_real = fill(0.5f0, 32, 32, 1, 1)
    d_g_fake_adv = fill(0.5f0, 32, 32, 1, 1)
    @test SinGAN.discriminator_loss(d_real, d_g_fake_adv) ≈ 0.5f0
    @inferred Float32 SinGAN.discriminator_loss(d_real, d_g_fake_adv)
end

@testset "generator_adv_loss" begin
    d_g_fake_adv = zeros(Float32, 32, 32, 1, 1)
    @test SinGAN.generator_adv_loss(d_g_fake_adv) ≈ 1f0
    @inferred Float32 SinGAN.generator_adv_loss(d_g_fake_adv)

    d_g_fake_adv = ones(Float32, 32, 32, 1, 1)
    @test SinGAN.generator_adv_loss(d_g_fake_adv) ≈ 0f0
    @inferred Float32 SinGAN.generator_adv_loss(d_g_fake_adv)
end

@testset "update_discriminator!" begin
    opt = ADAM(0.0005f0)
    dscr = SinGAN.build_single_discriminator(3, 32)
    real_img = ones(Float32, 2, 2, 3, 1)
    g_fake_adv = randn(Float32, 2, 2, 3, 1)
    @info SinGAN.update_discriminator!(opt, dscr, real_img, g_fake_adv)
    @inferred Float32 SinGAN.update_discriminator!(opt, dscr, real_img, g_fake_adv)
    # @code_warntype(SinGAN.update_discriminator!(opt, dscr, real_img, g_fake_adv))
end

@testset "update_generator_adv!" begin
    opt = ADAM(0.0005f0)
    dscr = SinGAN.build_single_discriminator(3, 4)
    gen = SinGAN.build_single_generator(3, 4, 1)
    prev_adv = SinGAN.resize_and_padding(randn(Float32, 2, 2, 3, 1), (2, 2), (4, 4))
    noise_adv = randn(Float32, 4, 4, 3, 1)
    @info SinGAN.update_generator_adv!(opt, dscr, gen, prev_adv, noise_adv)
    @inferred Float32 SinGAN.update_generator_adv!(opt, dscr, gen, prev_adv, noise_adv)
    # @code_warntype SinGAN.update_generator_adv!(opt, dscr, dscr, prev_adv, noise_adv)
end

@testset "update_generator_rec!" begin
    opt = ADAM(0.0005f0)
    gen = SinGAN.build_single_generator(3, 4, 1)
    real_img = ones(Float32, 2, 2, 3, 1)
    prev_rec = randn(Float32, 4, 4, 3, 1) # padded
    alpha = 10f0
    @info SinGAN.update_generator_rec!(opt, gen, real_img, prev_rec, alpha)
    @inferred Float32 SinGAN.update_generator_rec!(opt, gen, real_img, prev_rec, alpha)
    # @code_warntype SinGAN.update_generator_rec!(opt, gen, real_img, prev_rec, alpha)
end
