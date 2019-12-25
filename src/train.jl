"""
    loss functions
"""
function discriminator_loss(d_real, d_g_fake)
    real_loss = Flux.mse(1f0, d_real)
    fake_loss = Flux.mse(0f0, d_g_fake)
    return real_loss + fake_loss
end

generator_adv_loss(d_g_fake) = Flux.mse(1f0, d_g_fake)

generator_rec_loss(real_img, g_fake) = Flux.mse(real_img, g_fake)

"""
    train discriminator
"""

