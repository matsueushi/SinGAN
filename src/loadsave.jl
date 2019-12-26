using Printf

"""
    Image -> Array, Array -> Image
"""
function rgb_to_array(img_rgb)
    # Image -> 3D array
    # CHW -> WHC
    array = permutedims(Float32.(channelview(img_rgb)), (2, 3, 1))
    # [0, 1] -> [-1, 1]
    array = @. 2f0 * array - 1f0
    # 3D -> 4D
    return reshape(array, size(array)..., 1)
end

function save_array_as_image(path, array::AbstractArray{Float32,3})
    array = clamp.(array, -1f0, 1f0)
    img = permutedims(array, (3, 1, 2)) |> cpu
    img = (img .+ 1f0) ./ 2f0
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

scaled_real_savepath(n) = @sprintf("./output/real/step%03d.png", n)
fake_adv_savepath(n, epoch) = @sprintf("./output/fake/step%03d/adv/epoch%05d.png", n, epoch)
fake_rec_savepath(n, epoch) = @sprintf("./output/fake/step%03d/rec/epoch%05d.png", n, epoch)
discriminator_savepath(n) = @sprintf("./output/weights/dscr_step%03d.bson", n)
generator_savepath(n) = @sprintf("./output/weights/gen_step%03d.bson", n)

function save_training_loss(st, epoch, loss_dscr, loss_gen_adv, loss_gen_rec)
    path = @sprintf("./output/loss/step%03d_loss.csv", st)
    if epoch == 1
        open(path, "w") do io
            write(io, "epoch,loss_dscr,loss_gen_adv,loss_gen_rec\n")
        end
    end

    open(path, "a") do io
        write(io, @sprintf("%d,%06f,%06f,%06f\n", epoch, loss_dscr, loss_gen_adv, loss_gen_rec))
    end
end

function save_scaled_reals(real_img_p) 
    # save real images
    for (n, img) in enumerate(real_img_p)
        save_array_as_image(scaled_real_savepath(n), img[:, :, :, 1])
    end
end