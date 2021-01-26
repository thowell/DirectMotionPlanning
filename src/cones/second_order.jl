"""
- second-order (so)
    ||z1|| <= z2

A Semismooth Newton Method for Fast, Generic Convex Programming
    https://arxiv.org/abs/1705.00772

# todo: try backtracking linesearch for cone
    https://digital.lib.washington.edu/researchworks/bitstream/handle/1773/43279/Dueri_washington_0250E_19426.pdf?isAllowed=y&sequence=1
"""

# second-order cone
function κ_so(z)
    z1 = z[1:end-1]
    z2 = z[end]

    z_proj = zero(z)

    if norm(z1) <= z2
        z_proj = copy(z)
    elseif norm(z1) <= -z2
        z_proj = zero(z)
    else
        a = 0.5 * (1.0 + z2 / norm(z1))
        z_proj[1:end-1] = a * z1
        z_proj[end] = a * norm(z1)
    end

    return z_proj
end

# second-order cone Jacobian
function Jκ_so(z)
    z1 = z[1:end-1]
    z2 = z[end]
    m = length(z)

    if norm(z1) <= z2
        return Diagonal(ones(m))
    elseif norm(z1) <= -z2
        return Diagonal(zeros(m))
    else
        D = zeros(m, m)
        for i = 1:m
            if i < m
                D[i, i] = 0.5 + 0.5 * z2 / norm(z1) - 0.5 * z2 * ((z1[i])^2.0) / norm(z1)^3.0
            else
                D[i, i] = 0.5
            end
            for j = 1:m
                if j > i
                    if j < m
                        D[i, j] = -0.5 * z2 * z1[i] * z1[j] / norm(z1)^3.0
                        D[j, i] = -0.5 * z2 * z1[i] * z1[j] / norm(z1)^3.0
                    elseif j == m
                        D[i, j] = 0.5 * z1[i] / norm(z1)
                        D[j, i] = 0.5 * z1[i] / norm(z1)
                    end
                end
            end
        end
        return D
    end
e
