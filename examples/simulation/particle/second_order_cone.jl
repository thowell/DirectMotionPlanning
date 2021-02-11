
"""
    second-order cone
        assumes: z ∈ R³
"""
function κ_so(z)
    z1 = view(z, 2:3)
    z0 = z[1]

    z_proj = zero(z)
    status = false

    if norm(z1) <= z0
        z_proj = copy(z)
        if norm(z1) < z0
            status = true
        end
        # status = true
    elseif norm(z1) <= -z0
        z_proj = zero(z)
    else
        a = 0.5 * (1.0 + z0 / norm(z1))
        z_proj[1:end-1] = a * z1
        z_proj[end] = a * norm(z1)
    end

    return z_proj, status
end

function cone_product(z, s)
    [z' * s; z[1] * view(s, 2:3) + s[1] * view(z, 2:3)]
end
