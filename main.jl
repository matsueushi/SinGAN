using SinGAN

using FileIO
using Flux

function main()
    # hp = HyperParams()
    hp = load_hyperparams("hyperparams.json")
    # save_hyperparams("hyperparams.json", hp)
    dscrp, genp = setup_models(hp)
    @info dscrp
    @info genp

    img_name = "floral_shoppe.jpg"
    orig_rgb_img = load(img_name)
    orig_img = SinGAN.rgb_to_array(orig_rgb_img) |> gpu
    img_shapes = image_shapes(hp)
    real_img_p = SinGAN.build_image_pyramid(orig_img, img_shapes, img_shapes)
    # load_model_params!(dscrp, genp, Base.length(img_shapes))
    
    amplifiers, z_rec = train!(dscrp, genp, real_img_p, hp)
    generate_animation(genp, 0.1f0, 0.9f0, first(amplifiers), z_rec)
end

main()