function generate_animation(genp::GeneratorPyramid, α, β, amplifier, z_rec)
    z = build_zero_pyramid(z_rec, genp.noise_shapes)
    st = Base.length(genp.noise_shapes)

    z_rand = amplifier * randn_like(z_rec, size(z_rec))
    z_prev1 = @. 0.95f0 * z_rec + 0.05f0
    z_prev2 = z_rec

    for i in 1:100
        z_rand = amplifier * randn_like(z_rec, size(z_rec))
        diff_curr = @. β * (z_prev1 - z_prev2) + (1 - β) * z_rand
        z_curr = @. α * z_rec + (1 - α) * (z_prev1 + diff_curr)
        z_prev1 = z_curr
        z_prev2 = z_prev1

        z[1] = z_curr
        img = genp(z, st, false)
        save_array_as_image(animation_savepath(i), img)
    end
end