module SinGAN

using Adapt
using BSON: @load, @save
using CuArrays
using Flux
using Flux: mse, pullback, glorot_normal
using Flux.Optimise: update!
using JSON
using OrderedCollections
using Random
using Statistics

export DiscriminatorPyramid, GeneratorPyramid, NoiseConnection, HyperParams, image_shapes, setup_models,
        load_model_params!, train!, load_hyperparams, save_hyperparams, generate_animation

include("models.jl")
include("utils.jl")
include("train.jl")
include("loadsave.jl")
include("animation.jl")

end # module
