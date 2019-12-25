expand_dim(t::Tuple{Int64,Int64}) = (t..., 3, 1)

function zoom_image(x::Array{Float32, 4}, imsize::Tuple{Int64, Int64})
    itp = interpolate(x, (BSpline(Linear()), BSpline(Linear()), NoInterp(), NoInterp()))
    ss = Float32.(size(x))
    xs = LinRange{Float32}(1f0, ss[1], imsize[1])
    ys = LinRange{Float32}(1f0, ss[2], imsize[2])
    zs = LinRange{Float32}(1f0, ss[3], size(x)[3])
    ws = LinRange{Float32}(1f0, ss[4], size(x)[4])
    return [itp(x, y, z, w) for x in xs, y in ys, z in zs, w in ws]
end