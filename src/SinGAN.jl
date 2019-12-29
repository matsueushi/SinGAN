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

export DiscriminatorPyramid, GeneratorPyramid, NoiseConnection, HyperParams, image_shapes, setup_models,
        train!, load_hyperparams, save_hyperparams

include("models.jl")
include("utils.jl")
include("train.jl")
include("loadsave.jl")

end # module
