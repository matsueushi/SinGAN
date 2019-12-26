@testset "main_test" begin
    scale = 4/3
    min_size = (16, 16)
    image_size = (20, 20)
    image_shapes = SinGAN.size_pyramid(scale, min_size, image_size)
    
    n_stage = Base.length(image_shapes)
    n_layers = 3


    dscrp = DiscriminatorPyramid(n_stage, n_layers)
    genp = GeneratorPyramid(image_shapes, n_layers)

    orig_img = randn(Float32, SinGAN.expand_dim(image_size))
    real_img_p = gpu.(SinGAN.image_pyramid_generation(orig_img, image_shapes))

    max_epoch = 20
    reduce_lr_epoch = 16
    save_image_every_epoch = 500
    save_loss_every_epoch = 100

    # max_epoch = 2000
    # reduce_lr_epoch = 1600

    loop_dscr = 3
    loop_gen = 3

    lr_dscr = 5e-4
    lr_gen = 5e-4
    
    alpha = 50f0

    SinGAN.train!(dscrp, genp, real_img_p, 
        max_epoch, reduce_lr_epoch, save_image_every_epoch, save_loss_every_epoch,
        loop_dscr, loop_gen, lr_dscr, lr_gen, alpha)
end
