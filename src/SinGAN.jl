module SinGAN

using Adapt
using CuArrays
using Flux
using Flux: mse, pullback
using Flux.Optimise: update!
using Interpolations
using Random

export Discriminator, Generator

include("models.jl")
include("utils.jl")
include("train.jl")

end # module
