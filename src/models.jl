# Re-define leakyrelu function
# https://github.com/FluxML/Flux.jl/issues/963
myleakyrelu(x::Real, a = oftype(x / one(x), 0.01)) = max(a * x, x / one(x))

conv_block(in, out) = [
        Conv((3, 3), in => out; init = glorot_normal, pad = (1, 1)),
        BatchNorm(out),
        x->myleakyrelu.(x, 0.2f0)
    ]

function build_layers(n_layers, in_chs, conv_chs, out_chs, σ)
    layers = conv_block(in_chs, conv_chs)
    for _ in 1:n_layers - 2
        push!(layers, conv_block(conv_chs, conv_chs)...)
    end
    tail_layer = Conv((3, 3), conv_chs => out_chs, σ;
        init = glorot_normal, pad = (1, 1))
    push!(layers, tail_layer)
    return Chain(layers...)
end

"""
    DiscriminatorPyramid
"""
mutable struct DiscriminatorPyramid{T <: Tuple}
    chains::T
    DiscriminatorPyramid(xs...) = new{typeof(xs)}(xs)
end

build_single_discriminator(n_layers, conv_chs) = build_layers(n_layers, 3, conv_chs, 1, identity)

function DiscriminatorPyramid(n_stage::Integer, n_layers::Integer)
    ds = build_single_discriminator.(n_layers, channel_pyramid(n_stage))
    return DiscriminatorPyramid(gpu.(ds)...)
end

function DiscriminatorPyramid(image_shapes::Vector{Tuple{Int64,Int64}}, n_layers::Integer)
    DiscriminatorPyramid(Base.length(image_shapes), n_layers)
end

function Base.show(io::IO, d::DiscriminatorPyramid)
    print(io, "DiscriminatorPyramid(")
    join(io, d.chains, ", \n")
    print(io, ")")
end

"""
    NoiseConnection
"""
mutable struct NoiseConnection
    layers
    pad::Int64
end

@Flux.functor NoiseConnection

function (nc::NoiseConnection)(prev::T, noise::T) where {T <: AbstractArray{Float32,4}}
    pad = nc.pad
    raw_output = nc.layers(noise + prev)::T + prev
    return raw_output[1 + pad:end - pad, 1 + pad:end - pad, :, :]
end

function Base.show(io::IO, nc::NoiseConnection)
    print(io, "NoiseConnection(", nc.layers, ", ", nc.pad, ")")
end

"""
    GeneratorPyramid
"""
mutable struct GeneratorPyramid{T <: Tuple}
    image_shapes::Vector{Tuple{Int64,Int64}}
    noise_shapes::Vector{Tuple{Int64,Int64}}
    pad::Int64
    chains::T
    GeneratorPyramid(image_shapes, noise_shapes, pad, xs...) = new{typeof(xs)}(image_shapes, noise_shapes, pad, xs)
end

build_single_gen_layers(n_layers, conv_chs) = build_layers(n_layers, 3, conv_chs, 3, tanh)
build_single_generator(n_layers, conv_chs, pad) = NoiseConnection(build_single_gen_layers(n_layers, conv_chs), pad)

function GeneratorPyramid(image_shapes::Vector{Tuple{Int64,Int64}}, n_layers::Integer, pad::Integer = 5)
    n_stage = Base.length(image_shapes)
    # receptive field = 11, floor(11/2) = 5
    noise_shapes = [2 * pad .+ s for s in image_shapes]
    ds = build_single_generator.(n_layers, channel_pyramid(n_stage), pad)
    return GeneratorPyramid(image_shapes, noise_shapes, pad, gpu.(ds)...)
end

function Base.show(io::IO, d::GeneratorPyramid)
    print(io, "GeneratorPyramid(")
    print(io, d.image_shapes, ", ")
    print(io, d.noise_shapes, ", ")
    println(io, d.pad, ", ")
    join(io, d.chains, ", \n")
    print(io, ")")
end

function (genp::GeneratorPyramid)(xs::AbstractVector{T}, st::Integer, resize::Bool) where {T <: AbstractArray{Float32,4}}
    if st == 0
        zeros_shape = resize ? first(genp.noise_shapes) : first(genp.image_shapes)
        return zeros_like(T, expand_dim(zeros_shape...))
    end
    prev = genp(xs, st - 1, true)
    out = genp.chains[st](prev, xs[st])
    return resize ? resize_and_padding(out, genp.image_shapes[st + 1], genp.noise_shapes[st + 1]) : out
end

"""
    HyperParams
"""
mutable struct HyperParams
    scale::Float64                  # progression scale, > 1
    min_size_x::Int64               # minimal image width
    min_size_y::Int64               # minimal image height
    img_size_x::Int64               # output image width
    img_size_y::Int64               # output image height
    n_layers::Int64                 # number of conv layers
    max_epoch::Int64                # training epochs
    reduce_lr_epoch::Int64          # reduce learining rate after training `redule_lr_epoch` epochs
    save_image_every_epoch::Int64   # save generated image every `save_image_every_epoch` epoch
    save_loss_every_epoch::Int64    # save loss every `save_loss_every_epoch` epoch
    loop_dscr::Int64                # training steps par descriminator training epoch
    loop_gen::Int64                 # training steps par generator training epoch
    lr_dscr::Float64                # discriminator learining rate
    lr_gen::Float64                 # generator learning rate
    alpha::Float32                  # rec loss coefficient
    amplifier_init::Float32         # noise amplifier
    HyperParams() = new(4/3, 25, 25, 128, 128, 5, 2000, 1600, 500, 100, 3, 3, 5e-4, 5e-4, 50f0, 1f0)
end

show_dict(hp::HyperParams) = OrderedDict(string(nm) => getfield(hp, nm) for nm in fieldnames(HyperParams))
image_shapes(hp::HyperParams) = size_pyramid(hp.scale, (hp.min_size_x, hp.min_size_y), (hp.img_size_x, hp.img_size_y))

function setup_models(hp::HyperParams)
    img_shapes = image_shapes(hp)
    dscrp = DiscriminatorPyramid(img_shapes, hp.n_layers) |> gpu
    genp = GeneratorPyramid(img_shapes, hp.n_layers) |> gpu
    return dscrp, genp
end