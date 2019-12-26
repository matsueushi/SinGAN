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
    train
"""
function train!(dscrp::DiscriminatorPyramid, genp::GeneratorPyramid, real_img_p)
    @info dscrp
    @info genp
    loop_dscr = 3
    loop_gen = 3
    alpha = 50f0

    lr_dscr = 5e-4
    lr_gen = 5e-4
    opt_dscr = ADAM(lr_dscr, (0.5, 0.999))
    opt_gen = ADAM(lr_gen, (0.5, 0.999))

    max_epoch = 20
    # emax_epoch = 2000

    reduce_lr_epoch = 16
    # reduce_lr_epoch = 1600

    amplifier_init = 1f0
    amplifiers = [amplifier_init]

    # fixed noise for rec
    fixed_rec_noise = rec_vector_generation(first(real_img_p), genp.noise_shapes, amplifier_init)
    
    for st in 1:Base.length(genp.image_shapes)
        @info "Step $(st)"
        # reset optimizer
        opt_dscr.eta = lr_dscr
        opt_gen.eta = lr_gen
        opt_dscr.state = IdDict()
        opt_gen.state = IdDict()

        # calculate noise amplifier
        if st > 1
            g_fake_rec = genp(fixed_rec_noise[1:st-1], true)
            rmse = sqrt(mse(real_img_p[st], g_fake_rec))
            amp = rmse * amplifier_init
            push!(amplifiers, amp)
            @info "Noise amplifier: $(amp)"
            prev_rec = g_fake_rec
        else
            @info "Noise amplifier: $(amplifier_init)"
            prev_rec = zero(first(real_img_p)) 
        end

        for ep in 1:max_epoch
            # reduce learnint rate
            if ep == reduce_lr_epoch
                @info "Redule learning rate"
                opt_dscr.eta /= 10
                opt_gen.eta /= 10
            end

            # training
            # prev_rec is only used to infer array type

            # discriminator
            noise_adv = noise_vector_generation(prev_rec, genp.noise_shapes[1:st], amplifiers)
            g_fake_adv = genp(noise_adv, false)
            update_discriminator!(opt_dscr, dscrp.chains[st], real_img_p[st], g_fake_adv)

            # generator
            # adv
            
            noise_adv_full = noise_vector_generation(prev_rec, genp.noise_shapes[1:st], amplifiers)
            prev_adv = st == 1 ? zero(prev_rec) : genp(noise_adv_full[1:end-1], true)
            update_generator_adv!(opt_gen, dscrp.chains[st], genp.chains[st], prev_adv, last(noise_adv_full))

            # rec
            update_generator_rec!(opt_gen, genp.chains[st], real_img_p[st], prev_rec, alpha)            
        end
    
    end
end