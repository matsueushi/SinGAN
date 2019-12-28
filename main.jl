using SinGAN

using FileIO
using Flux

function main()

    scale = 4/3
    min_size = (16, 16)
    image_size = (20, 20)
    image_shapes = SinGAN.size_pyramid(scale, min_size, image_size)

    n_stage = Base.length(image_shapes)
    n_layers = 3

    dscrp = DiscriminatorPyramid(n_stage, n_layers) |> gpu
    genp = GeneratorPyramid(image_shapes, n_layers) |> gpu

    # orig_img = randn(Float32, SinGAN.expand_dim(image_size))
    orig_rgb_img = load("artwork.jpg")
    orig_img = SinGAN.rgb_to_array(orig_rgb_img) |> gpu

    real_img_p = SinGAN.build_image_pyramid(orig_img, image_shapes, image_shapes)

    max_epoch = 2000
    reduce_lr_epoch = 1600
    save_image_every_epoch = 500
    save_loss_every_epoch = 100

    # max_epoch = 2000
    # reduce_lr_epoch = 1600

    loop_dscr = 3
    loop_gen = 3

    lr_dscr = 5e-4
    lr_gen = 5e-4

    alpha = 50f0

    train!(dscrp, genp, real_img_p, 
        max_epoch, reduce_lr_epoch, save_image_every_epoch, save_loss_every_epoch,
        loop_dscr, loop_gen, lr_dscr, lr_gen, alpha)
end

main()