abstract type StageObjective <: Objective end

struct StageQuadratic <: StageObjective
    Q
    q
    R
    r
    T
end

g(obj::StageObjective, x, u, t) = 0.0

function objective(obj::StageObjective, x, u)
    T = obj.T
    J = 0.0

    for t = 1:T-1
        J += g(obj, x[t], u[t], t)
    end
    J += g(obj, x[T], nothing, T)

    return J
end
