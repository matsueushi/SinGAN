using Images

expand_dim(t::Tuple{Int64,Int64}) = (t..., 3, 1)

function zoom_pad_image(x::Array{Float32,4}, image_shape, noise_shape)
    # println(size(x), image_shape, noise_shape)
    x_large = mapslices(x->imresize(x, image_shape...), x; dims = (1, 2, 3))
    xx = zeros(Float32, expand_dim(noise_shape))
    pad1, pad2 = (noise_shape .- image_shape)[1:2] .÷ 2
    xx[1 + pad1:image_shape[1] + pad1, 1 + pad2:image_shape[2] + pad2, : , :] = x_large
    return xx
end

function zoom_pad_image(x::CuArray{Float32,4}, image_shape, noise_shape)
    return cu(zoom_pad_image(adapt(Array{Float32}, x), image_shape, noise_shape))
end

function build_image_pyramid(img::AbstractArray{Float32,4}, image_shapes, noise_shapes)
    return [zoom_pad_image(img, is, ns) for (is, ns) in zip(image_shapes, noise_shapes)]
end

build_zero_pyramid(xs, shapes) = [fill!(similar(xs, expand_dim(s)), 0f0) for s in shapes]

function build_noise_vector(xs, noise_shapes, amplifiers::Vector{Float32})
    return [amp * randn!(similar(xs, expand_dim(s))) for (s, amp) in zip(noise_shapes, amplifiers)]
end

function build_rec_vector(xs, noise_shapes, amplifier::Float32)
    v = build_zero_pyramid(xs, noise_shapes)
    randn!(v[1])
    v[1] *= amplifier
    return v
end