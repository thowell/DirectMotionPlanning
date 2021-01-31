"""
    - free cone
        x ∈ Rⁿ

    A Semismooth Newton Method for Fast, Generic Convex Programming
        https://arxiv.org/abs/1705.00772
"""

# free cone
function κ_free(z)
    z
end

# free cone Jacobian
function Jκ_free(z)
    p = ones(length(z))
    return Diagonal(p)
end
