expand_dim(t::Tuple{Int64,Int64}) = (t..., 3, 1)

function zoom_image(x::Array{Float32,4}, imsize::Tuple{Int64,Int64})
    itp = interpolate(x, (BSpline(Linear()), BSpline(Linear()), NoInterp(), NoInterp()))
    ss = Float32.(size(x))
    xs = LinRange{Float32}(1f0, ss[1], imsize[1])
    ys = LinRange{Float32}(1f0, ss[2], imsize[2])
    zs = LinRange{Float32}(1f0, ss[3], size(x)[3])
    ws = LinRange{Float32}(1f0, ss[4], size(x)[4])
    return [itp(x, y, z, w) for x in xs, y in ys, z in zs, w in ws]
end

function zoom_image(x::CuArray{Float32,4}, imsize::Tuple{Int64,Int64})
    return cu(zoom_image(adapt(Array{Float32}, x), imsize))
end

function image_pyramid_generation(img::Array{Float32,4}, image_shapes)
    return [zoom_image(img, s) for s in image_shapes]
end

similar_zero_pyramid(xs, shapes) = [fill!(similar(xs, expand_dim(s)), 0f0) for s in shapes]

function noise_vector_generation(xs, noise_shapes, amplifiers::Vector{Float32})
    return [amp * randn!(similar(xs, expand_dim(s))) for (s, amp) in zip(noise_shapes, amplifiers)]
end

function rec_vector_generation(xs, noise_shapes, amplifier::Float32)
    v = similar_zero_pyramid(xs, noise_shapes)
    randn!(v[1])
    v[1] *= amplifier
    return v
end