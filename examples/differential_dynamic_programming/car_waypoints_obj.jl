include_ddp()

# Model
include_model("car")

np = 2 * model.n + 2 * model.m + 2 * model.n

function fd(model::Car{Midpoint, FixedTime}, x, u, w, h, t)
	if t == 1
		θ = view(u, model.m .+ (1:np))
	else
		θ = view(x, model.m .+ (1:np))
	end

	return [view(x, 1:model.n) + h * f(model, view(x, 1:model.n) + 0.5 * h * f(model, view(x, 1:model.n), view(u, 1:model.m), w), view(u, 1:model.m), w);
		    θ]
end

n = [t == 1 ? model.n : (model.n + np) for t = 1:T]
m = [t == 1 ? (model.m + np) : model.m for t = 1:T]

# Time
T = 101
h = 0.05

# Reference trajectory
xx = range(0.0, stop = 1.0, length = T)
yy = 0.1 * sin.(2 * π * xx)
θθ = 0.2 * π * cos.(2 * π * xx)
plot(xx, yy, color = :black)
scatter!(xx, yy, color = :black)

p_ref = [[xx[t]; yy[t]] for t = 1:T]

# Initial conditions, controls, disturbances
x1 = [xx[1]; yy[1]; θθ[1]]#[0.0, 0.0, 0.0]
xT = [xx[T]; yy[T]; θθ[T]; zeros(np)]#[1.0, 0.0, 0.0] # goal state
ū = [t == 1 ? [1.0e-3 * randn(model.m); ones(np)] : 1.0e-3 * randn(model.m) for t = 1:T-1]
w = [zeros(model.d) for t = 1:T-1]

# Rollout
x̄ = rollout(model, x1, ū, w, h, T)

# Objective
obj = StageCosts([NonlinearCost() for t = 1:T], T)

function g(obj::StageCosts, x, u, t)
	T = obj.T

	J = 0.0

	# tracking
	if t == 1
		θ = view(u, model.m .+ (1:np))

		Q = Diagonal(θ[1:model.n])
		q = θ[model.n .+ (1:model.n)]
		R = Diagonal(θ[2 * model.n .+ (1:model.m)])
		r = θ[2 * model.n + model.m .+ (1:model.m)]

		xt = view(x, 1:model.n)
		ut = view(u, 1:model.m)

		J += xt' * Q * xt + q' * xt + ut' * R * ut + r' * ut
	elseif t > 1 && t < T
		θ = view(x, model.n .+ (1:np))

		Q = Diagonal(θ[1:model.n])
		q = θ[model.n .+ (1:model.n)]
		R = Diagonal(θ[2 * model.n .+ (1:model.m)])
		r = θ[2 * model.n + model.m .+ (1:model.m)]

		xt = view(x, 1:model.n)
		ut = view(u, 1:model.m)

		J += xt' * Q * xt + q' * xt + ut' * R * ut + r' * ut
	elseif t == T
		θ = view(x, model.n .+ (1:np))

		Q = Diagonal(θ[2 * model.n + 2 * model.m .+ (1:model.n)])
		q = θ[2 * model.n + 2 * model.m + model.n .+ (1:model.n)]

		xt = view(x, 1:model.n)

		J += xt' * Q * xt + q' * xt
	else
		nothing
	end

	return J
end

# Constraints
p = [t < T ? 2 * model.m + 2 : model.n for t = 1:T]
info_t = Dict(:p => p_ref, :ul => [-10.0; -10.0], :uu => [10.0; 10.0], :inequality => (1:2 * model.m))
info_T = Dict(:xT => xT)
con_set = [StageConstraint(p[t], t < T ? info_t : info_T) for t = 1:T]

function c!(c, cons::StageConstraints, x, u, t)
	T = cons.T
	p = cons.con[t].p

	if t < T
		ul = cons.con[t].info[:ul]
		uu = cons.con[t].info[:uu]
		c[1:(2 * model.m)] .= [ul - view(u, 1:model.m); view(u, 1:model.m) - uu]

		if t % 10 == 0
			pt = cons.con[t].info[:p]
			c[(2 * model.m) .+ (1:2)] .= x[1:2] - pt[t]
		end
	elseif t == T
		xT = cons.con[T].info[:xT]
		c[1:model.n] .= view(x, 1:model.n) - view(xT, 1:model.n)
	else
		nothing
	end
end

prob = problem_data(model, obj, con_set, copy(x̄), copy(ū), w, h, T,
	n = n, m = m)

# Solve
@time constrained_ddp_solve!(prob,
    max_iter = 1000, max_al_iter = 10,
	con_tol = 1.0e-4,
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
