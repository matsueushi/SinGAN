using Images

expand_dim(dim...) = (dim..., 3, 1)
zeros_like(T::Type, dims...) = fill!(similar(T, dims...), 0f0)
zeros_like(xs, dims...) = fill!(similar(xs, dims...), 0f0)
randn_like(T::Type, dims...) = randn!(similar(T, dims...))
randn_like(xs, dims...) where{T} = randn!(similar(xs, dims...))

function resize_and_padding(x::Array{Float32,4}, image_shape, padded_shape)
    # println(size(x), image_shape, padded_shape)
    x_large = mapslices(x->imresize(x, image_shape...), x; dims = (1, 2, 3))
    xx = zeros(Float32, expand_dim(padded_shape...))
    pad1, pad2 = (padded_shape .- image_shape)[1:2] .÷ 2
    xx[1 + pad1:image_shape[1] + pad1, 1 + pad2:image_shape[2] + pad2, : , :] = x_large
    return xx
end

function resize_and_padding(x::CuArray{Float32,4}, image_shape, padded_shape)
    return cu(resize_and_padding(adapt(Array{Float32}, x), image_shape, padded_shape))
end

function build_image_pyramid(img::AbstractArray{Float32,4}, image_shapes, noise_shapes)
    return map((is, ns) -> resize_and_padding(img, is, ns), image_shapes, noise_shapes)
end

build_zero_pyramid(xs, shapes) = map(s->zeros_like(xs, expand_dim(s...)), shapes)

function build_noise_vector(xs, noise_shapes, amplifiers::Vector{Float32})
    return map((s,a) -> a * randn_like(xs, expand_dim(s...)), noise_shapes, amplifiers)
end

function build_rec_vector(xs, noise_shapes, amplifier::Float32)
    v = build_zero_pyramid(xs, noise_shapes)
    randn!(v[1])
    v[1] *= amplifier
    return v
end