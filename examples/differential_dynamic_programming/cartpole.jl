include_ddp()

# Model
include_model("cartpole")
n, m, d = 4, 1, 4
model = Cartpole{Midpoint, FixedTime}(n, m, d, 1.0, 0.2, 0.5, -9.81) # flip world

n = model.n
m = model.m

# Time
T = 51
h = 0.05

# Initial conditions, controls, disturbances
x1 = [0.0, π, 0.0, 0.0]
xT = [0.0, 0.0, 0.0, 0.0] # goal state
ū = [1.0e-1 * ones(model.m) for t = 1:T-1]
w = [zeros(model.d) for t = 1:T-1]

# Rollout
x̄ = rollout(model, x1, ū, w, h, T)
# x̄ = linear_interpolation(x1, xT, T)

# Objective
Q = [(t < T ? Diagonal(1.0e-1 * ones(model.n))
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
x̄_nom, ū_nom = nominal_trajectory(prob)

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
visualize!(vis, model, x, Δt = h)

@save joinpath(@__DIR__, "trajectories/cartpole.jld2") x̄_nom ū_nom
