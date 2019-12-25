module SinGAN

using Adapt
using CuArrays
using Flux
using Interpolations
using Random

export Discriminator, Generator

include("models.jl")
include("utils.jl")

end # module
