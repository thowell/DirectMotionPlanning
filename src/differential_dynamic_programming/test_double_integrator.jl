include(joinpath(@__DIR__, "differential_dynamic_programming.jl"))

# Model
include_model("double_integrator")
n = model.n
m = model.m

# Time
T = 10
h = 1.0

# Initial conditions, controls, disturbances
x1 = rand(model.n)
ū = [rand(model.m) for t = 1:T-1]
w = [zeros(model.d) for t = 1:T-1]

# Rollout
x̄ = rollout(model, x1, ū, w, h, T)
# x̄ = linear_interpolation(x1, zeros(model.n), T)

# Objective
Q = Diagonal(ones(model.n))
R = Diagonal(ones(model.m))
q = zeros(model.n)
r = zeros(model.m)
obj = StageQuadratic(Q, q, R, r, T)

function g(obj::StageObjective, x, u, t)
    Q = obj.Q
    R = obj.R
    T = obj.T

    if t < T
        return x' * Q * x + u' * R * u
    elseif t == T
        return x' * Q * x
    else
        return 0.0
    end
end

# Solve
x̄, ū = solve(model, obj, x̄, ū, w, h, T)

# Visualize
using Plots
plot(hcat(x̄...)')
plot(hcat(ū..., ū[end])', linetype = :steppost)
