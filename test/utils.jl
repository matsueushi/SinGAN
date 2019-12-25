@testset "zoom_image" begin
    x = randn(Float32, 2, 2, 3, 1)
    @info x
    @info SinGAN.zoom_image(x, (3, 3))
end