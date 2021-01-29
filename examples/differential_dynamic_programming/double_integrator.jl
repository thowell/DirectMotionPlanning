include_ddp()

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

# Objective
Q = Diagonal(ones(model.n))
R = Diagonal(ones(model.m))

obj = StageQuadratic(Q, nothing, R, nothing, T)

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

@time x, u = solve(model, obj, copy(x̄), copy(ū), w, h, T)

# Visualize
using Plots
plot(hcat(x...)')
plot(hcat(u..., u[end])', linetype = :steppost)
