include_ddp()

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
Q = [(t < T ? Diagonal(1.0 * ones(model.n))
        : Diagonal(1000.0 * ones(model.n))) for t = 1:T]
R = Diagonal(1.0e-1 * ones(model.m))
obj = StageCosts([QuadraticCost(Q[t], nothing,
	t < T ? R : nothing, nothing) for t = 1:T], T)

function g(obj::StageCosts, x, u, t)
    T = obj.T
    if t < T
		Q = obj.cost[t].Q
		R = obj.cost[t].R
        return (x - xT)' * Q * (x - xT) + u' * R * u
    elseif t == T
		Q = obj.cost[t].Q
        return (x - xT)' * Q * (x - xT)
    else
        return 0.0
    end
end

# Problem
prob = problem_data(model, obj, copy(x̄), copy(ū), w, h, T)

# Solve
@time ddp_solve!(prob,
    max_iter = 100, verbose = true)

x, u = current_trajectory(prob)
x̄, ū = nominal_trajectory(prob)


# Visualize
using Plots
plot(hcat(x...)', label = "")
plot(hcat(u..., u[end])', linetype = :steppost)
