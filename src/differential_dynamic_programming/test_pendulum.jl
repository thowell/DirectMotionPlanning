include(joinpath(@__DIR__, "differential_dynamic_programming.jl"))

# Model
include_model("pendulum")
n = model.n
m = model.m

# Time
T = 26
h = 0.1

# Initial conditions, controls, disturbances
x1 = [0.0, 0.0]
xT = [π, 0.0] # goal state
ū = [1.0e-1 * rand(model.m) for t = 1:T-1]
w = [zeros(model.d) for t = 1:T-1]

# Rollout
x̄ = rollout(model, x1, ū, w, h, T)
# x̄ = linear_interpolation(x1, xT, T)

# Objective
Q = [(t < T ? Diagonal(1.0 * ones(model.n)) : Diagonal(10.0 * ones(model.n))) for t = 1:T]
R = Diagonal(1.0e-1 * ones(model.m))
q = [-2.0 * Q[t] * xT for t = 1:T]
r = zeros(model.m)
obj = StageQuadratic(Q, q, R, r, T)

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

# J = objective(obj, x̄, ū)
# # verbose && println("    initial cost: $J
# # ------------------")
#
# # compute initial derivatives
# fx, fu = dynamics_derivatives(model, x̄, ū, w, h, T)
# gx, gu, gxx, guu = objective_derivatives(obj, x̄, ū)
#
# K, _k, P, p, ΔV, Qx, Qu, Qxx, Quu, Qux = backward_pass(fx, fu, gx, gu, gxx, guu)
# x̄, ū, fx, fu, gx, gu, gxx, guu, J = forward_pass(model, obj, K, _k, x̄, ū, w, h, T, J)
# @show J
# x, u = rollout(model, K, _k, x̄, ū, w, h, T, α = 0.000125)
# J = objective(obj, x, u)
# x̄ = x
# ū = u

# x̄, ū, fx, fu, gx, gu, gxx, guu, J = forward_pass(model, obj, K, k, x̄, ū, w, h, T, J)
# grad_norm = norm(gradient(fx, fu, gx, gu, p))

# Solve
x̄, ū = solve(model, obj, x̄, ū, w, h, T,
    max_iter = 25)

# Visualize
using Plots
plot(hcat(x̄...)', label = "")
plot(hcat(ū..., ū[end])', linetype = :steppost)
