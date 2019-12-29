module SinGAN

using Adapt
using BSON: @load, @save
using CuArrays
using Flux
using Flux: mse, pullback, glorot_normal
using Flux.Optimise: update!
using Random

export DiscriminatorPyramid, GeneratorPyramid, NoiseConnection, train!

include("models.jl")
include("utils.jl")
include("train.jl")
include("loadsave.jl")

end # module
