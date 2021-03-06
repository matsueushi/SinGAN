"""
    loss functions
"""
function discriminator_loss(d_real::AbstractArray, d_g_fake_adv::AbstractArray)
    real_loss = mse(1f0, mean(d_real; dims = (1, 2)))
    fake_loss = mse(0f0, mean(d_g_fake_adv; dims = (1, 2)))
    return real_loss + fake_loss
end

generator_adv_loss(d_g_fake_adv::AbstractArray) = mse(1f0, mean(d_g_fake_adv; dims = (1, 2)))

generator_rec_loss(real_img::AbstractArray, g_fake_rec::AbstractArray) = mse(real_img, g_fake_rec)

"""
    update discriminator
"""
function update_discriminator!(opt, dscr, real_img, g_fake_adv)
    @eval Flux.istraining() = true
    ps = params(dscr)
    grad = gradient(ps) do
        discriminator_loss(dscr(real_img), dscr(g_fake_adv))
    end
    update!(opt, ps, grad)
    @eval Flux.istraining() = false
end

"""
    update generator
"""
# function update_generator_rec!(opt, gen, real_img, prev_rec, noise_rec, alpha)
#     @eval Flux.istraining() = true
#     ps = params(gen)
#     grad = gradient(ps) do
#         g_fake_rec = gen(prev_rec, noise_rec)
#         alpha * generator_rec_loss(real_img, g_fake_rec)
#     end
#     update!(opt, ps, grad)
#     @eval Flux.istraining() = false
# end

# function update_generator_adv!(opt, dscr, gen, prev_adv, noise_adv)
#     @eval Flux.istraining() = true
#     ps = params(gen)
#     grad = gradient(ps) do
#         d_g_fake_adv = dscr(gen(prev_adv, noise_adv))
#         generator_adv_loss(d_g_fake_adv)
#     end
#     update!(opt, ps, grad)
#     @eval Flux.istraining() = false
# end

function update_generator!(opt, dscr, gen, real_img, prev_rec, prev_adv, noise_rec, noise_adv, alpha)
    @eval Flux.istraining() = true
    ps = params(gen)
    grad = gradient(ps) do
        g_fake_rec = gen(prev_rec, noise_rec)
        d_g_fake_adv = dscr(gen(prev_adv, noise_adv))
        generator_adv_loss(d_g_fake_adv) + alpha * generator_rec_loss(real_img, g_fake_rec)
    end
    update!(opt, ps, grad)
    @eval Flux.istraining() = false
end

"""
    train
"""
function train_epoch!(opt_dscr, opt_gen, st, loop_dscr, loop_gen,
        dscr, genp, prev_rec, noise_rec, real_img, amplifiers, alpha)
    # training
    # prev_rec is only used to infer array type

    # discriminator
    foreach(1:loop_dscr) do _
        noise_adv = build_noise_pyramid(prev_rec, genp.noise_shapes[1:st], amplifiers)
        g_fake_adv = genp(noise_adv, st, false)
        # add noise to real
        # real_noise = 0.5f0 * amplifiers[st] * randn!(similar(real_img))
        # update_discriminator!(opt_dscr, dscr, real_img + real_noise, g_fake_adv)

        update_discriminator!(opt_dscr, dscr, real_img, g_fake_adv)
    end

    # generator
    foreach(1:loop_gen) do _
        # adv
        noise_adv = build_noise_pyramid(prev_rec, genp.noise_shapes[1:st], amplifiers)
        prev_adv = genp(noise_adv, st - 1, true)
        # update_generator_adv!(opt_gen, dscr, genp.chains[st], prev_adv, last(noise_adv))
        # rec
        # update_generator_rec!(opt_gen, genp.chains[st], real_img, prev_rec, noise_rec, alpha)
        update_generator!(opt_gen, dscr, genp.chains[st], real_img, prev_rec, prev_adv, noise_rec, last(noise_adv), alpha)
    end

    noise_adv = build_noise_pyramid(prev_rec, genp.noise_shapes[1:st], amplifiers)
    g_fake_adv = genp(noise_adv, st, false)
    loss_dscr = discriminator_loss(dscr(real_img), dscr(g_fake_adv))
    d_g_fake_adv = dscr(g_fake_adv)
    loss_gen_adv = generator_adv_loss(d_g_fake_adv)
    g_fake_rec = genp.chains[st](prev_rec, noise_rec)
    loss_gen_rec = generator_rec_loss(real_img, g_fake_rec) 

    return loss_dscr, loss_gen_adv, loss_gen_rec
end

function estimate_noise_amplifier(prev_rec::AbstractArray{Float32,4}, real_img::AbstractArray{Float32,4},
        pad::Integer, amplifier_init::Float32)
    prev_rec_crop = @view prev_rec[1 + pad:end - pad, 1 + pad:end - pad, :, :]
    rmse = sqrt(mse(real_img, prev_rec_crop))
    return rmse * amplifier_init
end

function train!(dscrp::DiscriminatorPyramid, genp::GeneratorPyramid, 
        real_img_p::Vector{T}, hp::HyperParams) where {T <: AbstractArray{Float32,4}}
    stages = Base.length(genp.image_shapes)
    generate_dirs(stages)
    save_scaled_reals(real_img_p)

    amplifiers = Float32[]

    # fixed noise for rec
    fixed_noise_rec = build_rec_pyramid(first(real_img_p), genp.noise_shapes, 1f0)
    fixed_noise_adv = similar(fixed_noise_rec)
    
    for st in 1:stages
        @info "Step $(st)"
        # reset optimizer
        opt_dscr = ADAM(hp.lr_dscr, (0.5, 0.999))
        opt_gen = ADAM(hp.lr_gen, (0.5, 0.999))

        # calculate noise amplifier
        prev_rec = genp(fixed_noise_rec, st - 1, true) # padded
        amp = st == 1 ? 1f0 : estimate_noise_amplifier(prev_rec, real_img_p[st], genp.pad, hp.amplifier_init)
        push!(amplifiers, amp)
        # add noise for adv 
        fixed_noise_adv[st] = amp * randn_like(prev_rec, expand_dim(genp.noise_shapes[st]...))

        save_noise_amplifiers(st, amp)
        @info "Noise amplifier = $(amp)"

        @time for ep in 1:hp.max_epoch
            # reduce learnint rate
            if ep == hp.reduce_lr_epoch
                @info "Reduce learning rate"
                opt_dscr.eta /= 10
                opt_gen.eta /= 10
            end

            loss_dscr, loss_gen_adv, loss_gen_rec =
                train_epoch!(opt_dscr, opt_gen, st, hp.loop_dscr, hp.loop_gen,
                    dscrp.chains[st], genp, prev_rec, fixed_noise_rec[st], real_img_p[st], amplifiers, hp.alpha)

            # save image/loss
            if ep == 1 || ep % hp.save_image_every_epoch == 0 || ep == hp.max_epoch
                save_generated_images(genp, fixed_noise_rec, fixed_noise_adv, st, ep)
            end

            if ep == 1 || ep % hp.save_loss_every_epoch == 0 || ep == hp.max_epoch
                @info "Epoch $(ep)" loss_dscr loss_gen_adv loss_gen_rec
                save_training_loss(st, ep, loss_dscr, loss_gen_adv, loss_gen_rec)
            end
        end

        save_model_params(dscrp, genp, st)
    end

    return amplifiers, first(fixed_noise_rec)
end