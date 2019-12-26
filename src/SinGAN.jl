module SinGAN

using Adapt
using CuArrays
using Flux
using Flux: mse, pullback
using Flux.Optimise: update!
using Random

export DiscriminatorPyramid, GeneratorPyramid, train!

include("models.jl")
include("utils.jl")
include("train.jl")
include("loadsave.jl")

end # module
