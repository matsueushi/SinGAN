using DelimitedFiles
using Printf

"""
    Image -> Array, Array -> Image
"""
function rgb_to_array(img_rgb)
    # Image -> 3D array
    # CHW -> WHC
    array = permutedims(Float32.(channelview(img_rgb)), (3, 2, 1))
    # [0, 1] -> [-1, 1]
    array = @. 2f0 * array - 1f0
    # 3D -> 4D
    return reshape(array, size(array)..., 1)
end

function save_array_as_image(path, array::AbstractArray{Float32,3})
    array = clamp.(array, -1f0, 1f0)
    img = permutedims(array, (3, 2, 1)) |> cpu
    img = @. (img + 1f0) / 2f0
    save(path, colorview(RGB, img))
end


"""
    Make directories
"""
function generate_dirs(max_step)
    dirs = ["./output/real",
            "./output/loss",
            "./output/weights",
            "./output/animation"]
    for s in 1:max_step
        push!(dirs, @sprintf("./output/fake/step%03d/adv", s))
        push!(dirs, @sprintf("./output/fake/step%03d/rec", s))
    end

    for d in dirs
        isdir(d) || mkpath(d)
    end
end

"""
    Load/Save paths
"""
scaled_real_savepath(n) = @sprintf("./output/real/step%03d.png", n)
fake_adv_savepath(n, epoch) = @sprintf("./output/fake/step%03d/adv/epoch%05d.png", n, epoch)
fake_rec_savepath(n, epoch) = @sprintf("./output/fake/step%03d/rec/epoch%05d.png", n, epoch)
discriminator_savepath(n) = @sprintf("./output/weights/dscr_step%03d.bson", n)
generator_savepath(n) = @sprintf("./output/weights/gen_step%03d.bson", n)
animation_savepath(n) = @sprintf("./output/animation/img%04d.png", n)

"""
    Save images
"""
function save_scaled_reals(real_img_p) 
    # save real images
    for (n, img) in enumerate(real_img_p)
        save_array_as_image(scaled_real_savepath(n), view(img, :, :, :, 1))
    end
end

function save_generated_images(genp::GeneratorPyramid, noise_rec::AbstractVector{T}, 
            noise_adv::AbstractVector{T}, st::Integer, ep::Integer) where {T <: AbstractArray{Float32,4}}
    g_fake_rec = genp(noise_rec, st, false)
    save_array_as_image(fake_rec_savepath(st, ep), view(g_fake_rec, :, :, :, 1))

    g_fake_adv = genp(noise_adv, st, false)
    save_array_as_image(fake_adv_savepath(st, ep), view(g_fake_adv, :, :, :, 1))
end

"""
    Load/Save hyperparameters
"""
function save_hyperparams(path, hp::HyperParams)
    open(path, "w") do io
        JSON.print(io, show_dict(hp), 4)
    end    
end

function load_hyperparams(path)
    hp = HyperParams()
    js = JSON.parsefile(path, inttype=Int64)
    foreach(js) do (k, v)
        sk = Symbol(k)
        setfield!(hp, sk, oftype(getfield(hp, sk), v))
    end
    return hp
end

"""
    Save params/result
"""
function save_training_loss(st, ep, loss_dscr, loss_gen_adv, loss_gen_rec)
    path = @sprintf("./output/loss/step%03d_loss.csv", st)
    if ep == 1
        open(path, "w") do io
            println(io, "epoch,loss_dscr,loss_gen_adv,loss_gen_rec")
        end
    end

    open(path, "a") do io
        println(io, ep, ",", loss_dscr, ",", loss_gen_adv, ",", loss_gen_rec)
    end
end

function save_noise_amplifiers(st, noise_amplifier)
    path = "./output/noise_amplifiers.csv"
    if st == 1
        open(path, "w") do io
            println(io, "noise_amplifiers")
        end
    end

    open(path, "a") do io
        println(io, noise_amplifier)
    end
end

function load_with_batchnorm!(chain::Chain, data)
    Flux.loadparams!(chain, params(data))
    for (lm, l) in zip(chain, data)
        if isa(l, BatchNorm)
            copyto!(lm.μ, l.μ)
            copyto!(lm.σ², l.σ²)
        end
    end
end

function load_model_params!(dscrp::DiscriminatorPyramid, genp::GeneratorPyramid, max_stage::Integer)
    for i in 1:max_stage
        @load discriminator_savepath(i) dscr
        load_with_batchnorm!(dscrp.chains[i], dscr |> gpu)

        @load generator_savepath(i) gen
        load_with_batchnorm!(genp.chains[i].layers, gen |> gpu)
    end
end

function save_model_params(dscrp::DiscriminatorPyramid, genp::GeneratorPyramid, st::Integer)
    dscr = dscrp.chains[st] |> cpu
    @save discriminator_savepath(st) dscr

    gen = genp.chains[st].layers |> cpu
    @save generator_savepath(st) gen
end