struct StageObjective
    g
end

function objective(obj::StageObjective, x, u)
    T = length(x)
    J = 0.0
    for t = 1:T-1
        J += obj.g[t](x[t], u[t])
    end
    J += obj.g[T](x[T])
    return J
end
