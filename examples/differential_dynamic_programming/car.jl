include_ddp()

# Model
include_model("car")
n = model.n
m = model.m

# Time
T = 101
h = 0.1

# Reference trajectory
xx = range(0.0, stop = 1.0, length = T)
yy = 0.1 * sin.(2 * π * xx)
θθ = 0.2 * π * cos.(2 * π * xx)
plot(xx, yy, color = :black)
scatter!(xx, yy, color = :black)

# Initial conditions, controls, disturbances
x1 = [xx[1]; yy[1]; θθ[1]]#[0.0, 0.0, 0.0]
xT = [xx[T]; yy[T]; θθ[T]]#[1.0, 0.0, 0.0] # goal state
ū = [1.0e-2 * rand(model.m) for t = 1:T-1]
w = [zeros(model.d) for t = 1:T-1]
# Rollout
x̄ = rollout(model, x1, ū, w, h, T)



# Objective
Q = [(t < T ? Diagonal([10.0; 10.0; 0.1])
        : Diagonal([10.0; 10.0; 0.1])) for t = 1:T]
q = [-2.0 * Q[t] * [xx[t]; yy[t]; θθ[t]]  for t = 1:T]

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

# Constraints
p = [t < T ? 2 * m : n for t = 1:T]
info_t = Dict(:ul => [-10.0; -10.0], :uu => [10.0; 10.0], :inequality => (1:2 * m))
info_T = Dict(:xT => xT)
con_set = [StageConstraint(p[t], t < T ? info_t : info_T) for t = 1:T]

function c!(c, cons::StageConstraints, x, u, t)
	T = cons.T
	p = cons.con[t].p

	if t < T
		ul = cons.con[t].info[:ul]
		uu = cons.con[t].info[:uu]
		c .= [ul - u; u - uu]
	elseif t == T
		xT = cons.con[T].info[:xT]
		c .= x - xT
	else
		nothing
	end
end

prob = problem_data(model, obj, con_set, copy(x̄), copy(ū), w, h, T)

# Solve
@time constrained_ddp_solve!(prob,
    max_iter = 1000, max_al_iter = 7,
	con_tol = 1.0e-3,
	ρ_init = 1.0, ρ_scale = 10.0)

x, u = current_trajectory(prob)
x̄, ū = nominal_trajectory(prob)

# Visualize
# using Plots
# plot(π * ones(T),
#     width = 2.0, color = :black, linestyle = :dash)
# plot!(hcat(x...)', width = 2.0, label = "")
#
# plot(hcat([con_set[1].info[:ul] for t = 1:T]...)',
#     width = 2.0, color = :black, label = "")
# plot!(hcat([con_set[1].info[:uu] for t = 1:T]...)',
#     width = 2.0, color = :black, label = "")
# plot!(hcat(u..., u[end])',
#     width = 2.0, linetype = :steppost,
# 	label = "", color = :orange)

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
# open(vis)
visualize!(vis, model, x, Δt = h)

plot(xx, yy, color = :black, width = 2.0)
plot!([xt[1] for xt in x], [xt[2] for xt in x], color = :cyan, width = 1.0)
