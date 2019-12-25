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
    image_shapes::Vector{Tuple{Int64, Int64}}
    noise_shapes::Vector{Tuple{Int64, Int64}}
    pad::Int64
    chains::T
    Generator(image_shapes, noise_shapes, pad, xs...) = new{typeof(xs)}(image_shapes, noise_shapes, pad, gpu.(xs))
end

build_single_generator(n_layers, conv_chs) = build_layer(n_layers, 3, conv_chs, 3, tanh)

function Generator(image_shapes, n_layers::Integer; pad::Integer = 5)
    n_stage = Base.length(image_shapes)
    # receptive field = 11, floor(11/2) = 5
    noise_shapes = [2 * pad .+ s for s in image_shapes]
    ds = build_single_generator.(n_layers, channel_pyramid(n_stage))
    return Generator(image_shapes, noise_shapes, pad, ds...)
end

function Base.show(io::IO, d::Generator)
    print(io, "Generator(")
    print(io, d.image_shapes, ", ")
    print(io, d.noise_shapes, ", ")
    println(io, d.pad, ", ")
    join(io, d.chains, ", \n")
    print(io, ")")
end

function generate(chains::Chain, img::T, noise::T, pad::Integer) where {T<:AbstractArray{Float32,4}}
    y = zero(noise)
    y[1 + pad:end-pad, 1 + pad:end-pad, :, :] = img
    y += noise
    return chains(y)[1 + pad:end-pad, 1 + pad:end-pad, :, :] + img
end

# adv
function (gen::Generator)(xs::Vector{T}, resize::Bool) where {T<:AbstractArray{Float32,4}}
    img = fill!(similar(first(xs), expand_dim(first(gen.image_shapes))), 0f0)
    n = Base.length(xs)
    for (i, x) in enumerate(xs)
        img = generate(gen.chains[i], img, x, gen.pad)
        if i != n || (i == n && resize)
            img = adapt(img, zoom_image(adapt(Array, img), gen.image_shapes[i + 1]))
        end
    end
    return img
end

similar_zero_pyramid(xs, shapes::Vector{Tuple{Int64,Int64}}) = [fill!(similar(xs, expand_dim(s)), 0f0) for s in shapes]

function noise_vector_generation(xs, noise_shapes::Vector{Tuple{Int64,Int64}}, amplifiers::Vector{Float32})
    return [amp * randn!(similar(xs, expand_dim(s))) for (s, amp) in zip(noise_shapes, amplifiers)]
end

function rec_vector_generation(xs, noise_shapes::Vector{Tuple{Int64,Int64}}, amplifier::Float32)
    v = similar_zero_pyramid(xs, noise_shapes)
    randn!(v[1])
    v[1] *= amplifier
    return v
end