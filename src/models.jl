function size_pyramid(scale, min_size, image_size)
    current_size = min_size 
    pyramid = Vector{Tuple{Int64,Int64}}()
    for i in 1:100
        push!(pyramid, current_size)
        current_size == image_size && break
        current_size = @. floor(min_size * scale^i)
        current_size = min.(current_size, image_size)
    end
    return pyramid
end

channel_pyramid(n_stage) = min.([32 * 2^(floor(Int64, s/4)) for s in 1:n_stage], 128)

# Re-define leakyrelu function
# https://github.com/FluxML/Flux.jl/issues/963
myleakyrelu(x::Real, a = oftype(x / one(x), 0.01)) = max(a * x, x / one(x))

conv_block(in, out) = [
        Conv((3, 3), in => out; init = Flux.glorot_normal, pad = (1, 1)),
        BatchNorm(out),
        x->myleakyrelu.(x, 0.2f0)
    ]

function build_layer(n_layers, in_chs, conv_chs, out_chs, σ)
    layers = conv_block(in_chs, conv_chs)
    for _ in 1:n_layers-2
        push!(layers, conv_block(conv_chs, conv_chs)...)
    end
    tail_layer = Conv((3, 3), conv_chs => out_chs, σ;
        init = Flux.glorot_normal, pad = (1, 1))
    push!(layers, tail_layer)
    return Chain(layers...)
end

"""
    DiscriminatorPyramid
"""
mutable struct DiscriminatorPyramid{T<:Tuple}
    chains::T
    DiscriminatorPyramid(xs...) = new{typeof(xs)}(xs)
end

build_single_discriminator(n_layers, conv_chs) = build_layer(n_layers, 3, conv_chs, 1, identity)

function DiscriminatorPyramid(n_stage::Integer, n_layers::Integer)
    ds = build_single_discriminator.(n_layers, channel_pyramid(n_stage))
    return DiscriminatorPyramid(gpu.(ds)...)
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

# adv connection
function (nc::NoiseConnection)(prev, noise)
    pad = nc.pad
    raw_output = nc.layers(noise + prev) + prev
    return raw_output[1 + pad:end-pad, 1 + pad:end-pad, :, :]
end

# rec connection
function (nc::NoiseConnection)(prev)
    pad = nc.pad
    return nc.layers(prev)[1 + pad:end-pad, 1 + pad:end-pad, :, :]
end

function Base.show(io::IO, nc::NoiseConnection)
    print(io, "NoiseConnection(", nc.layers, ", ", nc.pad, ")")
end

"""
    GeneratorPyramid
"""
mutable struct GeneratorPyramid{T<:Tuple}
    image_shapes::Vector{Tuple{Int64, Int64}}
    noise_shapes::Vector{Tuple{Int64, Int64}}
    pad::Int64
    chains::T
    GeneratorPyramid(image_shapes, noise_shapes, pad, xs...) = new{typeof(xs)}(image_shapes, noise_shapes, pad, xs)
end

build_single_gen_layers(n_layers, conv_chs) = build_layer(n_layers, 3, conv_chs, 3, tanh)
build_single_generator(n_layers, conv_chs, pad) = NoiseConnection(build_single_gen_layers(n_layers, conv_chs), pad)

function GeneratorPyramid(image_shapes, n_layers::Integer; pad::Integer = 5)
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

function (genp::GeneratorPyramid)(xs::AbstractArray{T}, resize::Bool) where {T<:AbstractArray{Float32,4}}
    st = Base.length(xs)
    if st == 0
        if resize
            return zeros_like(T, expand_dim(first(genp.noise_shapes)))
        else
            return zeros_like(T, expand_dim(first(genp.image_shapes)))
        end
    end

    xs_prev = @view xs[1:end - 1]
    prev = genp(xs_prev, true)
    out = genp.chains[st](prev, last(xs))
    if resize
        return resize_and_padding(out, genp.image_shapes[st + 1], genp.noise_shapes[st + 1])
    else
        return out
    end
end