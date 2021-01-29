using BenchmarkTools
include(joinpath(@__DIR__, "differential_dynamic_programming.jl"))

# Model
include_model("cartpole")
n = model.n
m = model.m

# Time
T = 31
h = 0.1

# Initial conditions, controls, disturbances
x1 = [0.0, 0.0, 0.0, 0.0]
xT = [0.0, π, 0.0, 0.0] # goal state
ū = [1.0e-1 * ones(model.m) for t = 1:T-1]
w = [zeros(model.d) for t = 1:T-1]

# Rollout
x̄ = rollout(model, x1, ū, w, h, T)
# x̄ = linear_interpolation(x1, xT, T)

# Objective
Q = [(t < T ? Diagonal(1.0 * ones(model.n))
    : Diagonal(100.0 * ones(model.n))) for t = 1:T]
R = Diagonal(1.0e-1 * ones(model.m))
obj = StageQuadratic(Q, nothing, R, nothing, T)

function g(obj::StageQuadratic, x, u, t)
    Q = obj.Q[t]
    R = obj.R
    T = obj.T

    if t < T
        return (x - xT)' * Q * (x - xT) + u' * R * u
    elseif t == T
        return (x - xT)' * Q * (x - xT)
    else
        return 0.0
    end
end

# Solve
@time x, u = solve(model, obj, copy(x̄), copy(ū), w, h, T,
    max_iter = 500, verbose = true)

"""
    benchmark times
    1.97s
    2.20s

"""
# Visualize
using Plots
plot(hcat(x...)', label = "")
plot(hcat(u..., u[end])', linetype = :steppost)
