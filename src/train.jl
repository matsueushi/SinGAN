"""
    loss functions
"""
function discriminator_loss(d_real, d_g_fake_adv)
    real_loss = mse(1f0, d_real)
    fake_loss = mse(0f0, d_g_fake_adv)
    return real_loss + fake_loss
end

generator_adv_loss(d_g_fake_adv) = mse(1f0, d_g_fake_adv)

generator_rec_loss(real_img, g_fake_rec) = mse(real_img, g_fake_rec)

"""
    update discriminator
"""
function update_discriminator!(opt, dscr, real_img, g_fake_adv)
    @eval Flux.istraining() = true
    ps = params(dscr)
    loss, back = pullback(ps) do
        discriminator_loss(dscr(real_img), dscr(g_fake_adv))
    end
    grad = back(1f0)
    update!(opt, ps, grad)
    @eval Flux.istraining() = false
    return loss
end

"""
    update generator
"""
function update_generator_adv!(opt, dscr, gen, prev_adv, noise_adv)
    @eval Flux.istraining() = true
    ps = params(gen)
    loss, back = pullback(ps) do
        d_g_fake_adv = dscr(gen(prev_adv, noise_adv))
        generator_adv_loss(d_g_fake_adv)
    end
    grad = back(1f0)
    update!(opt, ps, grad)
    @eval Flux.istraining() = false
    return loss
end

function update_generator_rec!(opt, gen, real_img, prev_rec, alpha)
    @eval Flux.istraining() = true
    ps = params(gen)
    loss, back = pullback(ps) do
        g_fake_rec = gen(prev_rec)
        alpha * generator_rec_loss(real_img, g_fake_rec)
    end
    grad = back(1f0)
    update!(opt, ps, grad)
    @eval Flux.istraining() = false
    return loss
end

"""
    train discriminator
"""
function train_discriminator!(opt, dscr, genp::GeneratorPyramid, real_img, noise_adv)
    g_fake_adv = genp(noise_adv, false)
    loss = update_discriminator!(opt, dscr, real_img, g_fake_adv)
    return loss
end

"""
    train generator
"""


"""
    train
"""