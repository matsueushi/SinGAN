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

    SinGAN.train!(dscrp, genp, real_img_p) 
    # @code_warntype SinGAN.train!(dscrp, genp, real_img_p) 
end
