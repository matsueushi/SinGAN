"""
    loss functions
"""
function discriminator_loss(d_real, d_g_fake)
    real_loss = mse(1f0, d_real)
    fake_loss = mse(0f0, d_g_fake)
    return real_loss + fake_loss
end

generator_adv_loss(d_g_fake) = mse(1f0, d_g_fake)

generator_rec_loss(real_img, g_fake) = mse(real_img, g_fake)

"""
    update discriminator
"""
function update_discriminator!(opt, dscr, real_img, g_fake)::Float32
    @eval Flux.istraining() = true
    ps = params(dscr)
    loss, back = pullback(ps) do
        discriminator_loss(dscr(real_img), dscr(g_fake))
    end
    grad = back(1f0)
    update!(opt, ps, grad)
    @eval Flux.istraining() = false
    return loss
end

"""
    update generator
"""
function update_generator!(opt, nc_gen, dscr, real_img, prev, noise, alpha)::Float32
    @eval Flux.istraining() = true
    ps = params(nc_gen)
    loss, back = pullback(ps) do
        g_fake = nc_gen(prev, noise)
        d_g_fake = dscr(g_fake)
        generator_adv_loss(d_g_fake) + alpha * generator_rec_loss(real_img, g_fake)
    end
    grad = back(1f0)
    update!(opt, ps, grad)
    @eval Flux.istraining() = false
    return loss
end

"""
    train discriminator
"""
function train_discriminator()
    

end


"""
    train generator
"""