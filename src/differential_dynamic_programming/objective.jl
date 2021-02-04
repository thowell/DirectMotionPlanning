abstract type StageCost end

struct StageCosts <: Objective
    cost::Vector{StageCost}
    T::Int
end

g(obj::StageCosts, x, u, t) = 0.0

function objective(obj::StageCosts, x, u)
    T = obj.T
    J = 0.0
    for t = 1:T-1
        J += g(obj, x[t], u[t], t)
    end
    J += g(obj, x[T], nothing, T)
    return J
end

function objective(data::ModelData; mode = :nominal)
    if mode == :nominal
        return objective(data.obj, data.x̄, data.ū)
    elseif mode == :current
        return objective(data.obj, data.x, data.u)
    end
end

"""
    quadratic stage cost
"""
struct QuadraticCost <: StageCost
    Q
    q
    R
    r
end
