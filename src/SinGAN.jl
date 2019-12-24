module SinGAN

using Adapt
using CuArrays
using Flux
using Interpolations

export Discriminator, Generator

include("models.jl")
include("utils.jl")

end # module
