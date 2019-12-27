using Interpolations
using Images

expand_dim(t::Tuple{Int64,Int64}) = (t..., 3, 1)

function zoom_pad_image(x::AbstractArray{Float32,4}, imsize::Tuple{Int64,Int64})
    itp = interpolate(x, (BSpline(Linear()), BSpline(Linear()), NoInterp(), NoInterp()))
    ss = Float32.(size(x))
    xs = LinRange{Float32}(1f0, ss[1], imsize[1])
    ys = LinRange{Float32}(1f0, ss[2], imsize[2])
    zs = LinRange{Float32}(1f0, ss[3], size(x)[3])
    ws = LinRange{Float32}(1f0, ss[4], size(x)[4])
    xx = zero(x)
    for i in xs, j in ys, k in zs, l in ws
        xx[i, j, k, l] = itp(i, j, k, l)
    end
    return xx
end

function zoom_pad_image(x::CuArray{Float32,4}, imsize::Tuple{Int64,Int64})
    return cu(zoom_pad_image(adapt(Array{Float32}, x), imsize))
end

function image_pyramid_generation(img::AbstractArray{Float32,4}, shapes)
    return [zoom_pad_image(img, s) for s in shapes]
end

zero_pyramid_generation(xs, shapes) = [fill!(similar(xs, expand_dim(s)), 0f0) for s in shapes]

function noise_vector_generation(xs, noise_shapes, amplifiers::Vector{Float32})
    return [amp * randn!(similar(xs, expand_dim(s))) for (s, amp) in zip(noise_shapes, amplifiers)]
end

function rec_vector_generation(xs, noise_shapes, amplifier::Float32)
    v = similar_zero_pyramid(xs, noise_shapes)
    randn!(v[1])
    v[1] *= amplifier
    return v
end