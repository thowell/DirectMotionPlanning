"""
    - nonnegative orthant (no)
        x >= 0

    A Semismooth Newton Method for Fast, Generic Convex Programming
        https://arxiv.org/abs/1705.00772
"""

# nonnegative orthant cone
function κ_no(z)
    max.(0.0, z)
end

# nonnegative orthant cone Jacobian
function Jκ_no(z)
    p = zero(z)
    for (i, pp) in enumerate(z)
        if pp >= 0.0
            p[i] = 1.0
        end
    end
    return Diagonal(p)
end
