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
    Discriminator
"""
mutable struct Discriminator{T<:Tuple}
    chains::T
    Discriminator(xs...) = new{typeof(xs)}(gpu.(xs))
end

build_single_discriminator(n_layers, conv_chs) = build_layer(n_layers, 3, conv_chs, 1, identity)

function Discriminator(n_stage::Integer, n_layers::Integer)
    ds = build_single_discriminator.(n_layers, channel_pyramid(n_stage))
    return Discriminator(ds...)
end

function Base.show(io::IO, d::Discriminator)
    print(io, "Discriminator(")
    join(io, d.chains, ", \n")
    print(io, ")")
end

"""
    NoiseConnection
"""
mutable struct NoiseConnection
    layers
    prev
end

@Flux.functor NoiseConnection

function (skip::NoiseConnection)(input)
    skip.layers(input + prev) + prev
end

function Base.show(io::IO, b::NoiseConnection)
    print(io, "NoiseConnection(", b.layers, ")")
end

"""
    Generator
"""
mutable struct Generator{T<:Tuple}
    pyramid::Vector{Tuple{Int64, Int64}}
    chains::T
    Generator(pyramid, xs...) = new{typeof(xs)}(pyramid, gpu.(xs))
end

build_single_generator(n_layers, conv_chs) = build_layer(n_layers, 3, conv_chs, 3, tanh)

function Generator(pyramid, n_layers::Integer)
    n_stage = Base.length(pyramid)
    ds = build_single_generator.(n_layers, channel_pyramid(n_stage))
    return Generator(pyramid, ds...)
end

function Base.show(io::IO, d::Generator)
    print(io, "Generator(")
    join(io, d.chains, ", \n")
    print(io, ")")
end

# rec
function (gen::Generator)(x, stage::Integer, resize::Bool)
    for (i, ch) in enumerate(gen.chains[1:stage])
        x = ch(x) + x
        if (i != stage) || ((i != Base.length(gen.chains)) && resize)
            x = zoom_image(x |> cpu, gen.pyramid[i + 1]) |> gpu
        end
    end
    return x
end

# adv
function (gen::Generator)(x, adv_noise, stage::Integer, resize::Bool)
    @assert Base.length(adv_noise) == stage - 1
    for (i, ch) in enumerate(gen.chains[1:stage])
        x = i == 1 ? ch(x) + x : ch(x + adv_noise[i - 1]) + x
        if (i != stage) || ((i != Base.length(gen.chains)) && resize)
            x = zoom_image(x |> cpu, gen.pyramid[i + 1]) |> gpu
        end
    end
    return x
end