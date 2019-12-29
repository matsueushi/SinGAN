using Images

"""
    expand dimensions, zeros, randn
"""
expand_dim(dim...) = (dim..., 3, 1)
zeros_like(T::Type, dims...) = fill!(similar(T, dims...), 0f0)
zeros_like(xs::AbstractArray, dims...) = fill!(similar(xs, dims...), 0f0)
randn_like(T::Type, dims...) = randn!(similar(T, dims...))
randn_like(xs::AbstractArray, dims...) where {T} = randn!(similar(xs, dims...))

"""
    size
"""
function size_pyramid(scale, min_size, image_size)
    current_size = min_size 
    pyramid = Vector{Tuple{Int64,Int64}}()
    for i in 1:100
        push!(pyramid, current_size)
        current_size == image_size && break
        current_size = @. floor(Int64, min_size * scale^i)
        current_size = min.(current_size, image_size)
    end
    return pyramid
end

channel_pyramid(n_stage) = min.(map(s->32 * 2^(floor(Int64, s / 4)), 1:n_stage), 128)

"""
    resize
"""
function resize_and_padding(x::Array{Float32,4}, 
            image_shape::Tuple{Int64,Int64}, padded_shape::Tuple{Int64,Int64})
    # println(size(x), image_shape, padded_shape)
    x_large = imresize(view(x, :, :, :, 1), image_shape...)
    xx = zeros(Float32, expand_dim(padded_shape...))
    pad1, pad2 = (@. (padded_shape - image_shape) ÷ 2)[1:2]
    xx[1 + pad1:image_shape[1] + pad1, 1 + pad2:image_shape[2] + pad2, : , 1] = x_large
    return xx
end

function resize_and_padding(x::CuArray{Float32,4}, 
            image_shape::Tuple{Int64,Int64}, padded_shape::Tuple{Int64,Int64})
    return cu(resize_and_padding(adapt(Array{Float32}, x), image_shape, padded_shape))
end

"""
    image pyramid
"""
function build_image_pyramid(img::AbstractArray{Float32,4}, 
            image_shapes::Vector{Tuple{Int64,Int64}}, noise_shapes::Vector{Tuple{Int64,Int64}})
    return map((is, ns)->resize_and_padding(img, is, ns), image_shapes, noise_shapes)
end

function build_zero_pyramid(xs::AbstractArray{Float32,4}, shapes::Vector{Tuple{Int64,Int64}})
    return map(s->zeros_like(xs, expand_dim(s...)), shapes)
end

"""
    noise pyramid
"""
function build_noise_pyramid(xs::AbstractArray{Float32,4}, shapes::Vector{Tuple{Int64,Int64}}, amplifiers::Vector{Float32})
    return map((s, a)->a * randn_like(xs, expand_dim(s...)), shapes, amplifiers)
end

function build_rec_pyramid(xs::AbstractArray{Float32,4}, shapes::Vector{Tuple{Int64,Int64}}, amplifier::Float32)
    v = build_zero_pyramid(xs, shapes)
    randn!(v[1])
    v[1] *= amplifier
    return v
end