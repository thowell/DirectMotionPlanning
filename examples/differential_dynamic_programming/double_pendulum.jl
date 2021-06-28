include_ddp()

# Model
include_model("double_pendulum")
n = model.n
m = model.m

# Time
T = 101
h = 0.05

# Initial conditions, controls, disturbances
x1 = [0.0, 0.0, 0.0, 0.0]
xT = [π, 0.0, 0.0, 0.0] # goal state
ū = [[0.0; 1.0e-3 * randn(1)[1]]  for t = 1:T-1]
w = [zeros(model.d) for t = 1:T-1]

# Rollout
x̄ = rollout(model, x1, ū, w, h, T)
# x̄ = linear_interpolation(x1, xT, T)
plot(hcat(x̄...)')

# Objective
Q = [(t < T ? Diagonal(1.0e-3 * ones(model.n))
        : Diagonal(100.0 * ones(model.n))) for t = 1:T]
q = [-2.0 * Q[t] * xT for t = 1:T]

R = [Diagonal(1.0e-3 * ones(model.m)) for t = 1:T-1]
r = [zeros(model.m) for t = 1:T-1]

obj = StageCosts([QuadraticCost(Q[t], q[t],
	t < T ? R[t] : nothing, t < T ? r[t] : nothing) for t = 1:T], T)

function g(obj::StageCosts, x, u, t)
	T = obj.T
    if t < T
		Q = obj.cost[t].Q
		q = obj.cost[t].q
	    R = obj.cost[t].R
		r = obj.cost[t].r
        return x' * Q * x + q' * x + u' * R * u + r' * u
    elseif t == T
		Q = obj.cost[T].Q
		q = obj.cost[T].q
        return x' * Q * x + q' * x
    else
        return 0.0
    end
end

# Problem
prob = problem_data(model, obj, copy(x̄), copy(ū), w, h, T)

# Solve
@time ddp_solve!(prob,
    max_iter = 1000, verbose = true)

x, u = current_trajectory(prob)
x̄, ū = nominal_trajectory(prob)


# Visualize
using Plots
plot(π * ones(T),
    width = 2.0, color = :black, linestyle = :dash)
plot!(hcat(x...)', width = 2.0, label = "")
plot(hcat(u..., u[end])',
    width = 2.0, linetype = :steppost)

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
# open(vis)
visualize!(vis, model, x, Δt = h)
