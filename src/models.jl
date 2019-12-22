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

build_single_discriminator(n_layers, conv_chs) = build_layer(n_layers, 3, conv_chs, 1, identity)
build_single_generator(n_layers, conv_chs) = build_layer(n_layers, 3, conv_chs, 3, tanh)


