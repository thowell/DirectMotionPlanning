using Random, Plots
include_ddp()

# Model
include_model("satellite")

# Time
T = 26
h = 0.1
tf = h * (T - 1)

# Initial conditions, controls, disturbances
mrp1 = MRP(RotXYZ(0.0, 0.0, 0.0))
mrpT = MRP(RotXYZ(0.5 * π, 0.5 * π, 0.0))

x1 = [mrp1.x, mrp1.y, mrp1.z, 0.0, 0.0, 0.0]
xT = [mrpT.x, mrpT.y, mrpT.z, 0.0, 0.0, 0.0]

ū = [1.0e-1 * ones(model.m) for t = 1:T-1]
w = [zeros(model.d) for t = 1:T-1]

# Rollout
x̄ = rollout(model, x1, ū, w, h, T)
# x̄ = linear_interpolation(x1, xT, T)

# Objective
Q = [(t < T ? Diagonal(h * [1.0; 1.0; 1.0; 1.0; 1.0; 1.0])
        : Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0])) for t = 1:T]
q = [-2.0 * Q[t] * xT for t = 1:T]

R = [h * Diagonal(1.0e-1 * ones(model.m)) for t = 1:T-1]
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
p_con = [t == T ? model.n : 0 for t = 1:T]

info_t = Dict()
info_T = Dict(:xT => xT)

con_set = [StageConstraint(p_con[t], t < T ? info_t : info_T) for t = 1:T]

function c!(c, cons::StageConstraints, x, u, t)
	T = cons.T
	n = model.n

	if t == T
		c[1:n] .= x - cons.con[T].info[:xT]
	end
end

prob = problem_data(model, obj, con_set, copy(x̄), copy(ū), w, h, T)

# Solve
@time constrained_ddp_solve!(prob,
    max_iter = 1000, max_al_iter = 10,
	ρ_init = 1.0, ρ_scale = 10.0,
	con_tol = 1.0e-3,
	cache = false)

x, u = current_trajectory(prob)
x̄, ū = nominal_trajectory(prob)

# Visualize
plot(π * ones(T),
    width = 2.0, color = :black, linestyle = :dash)
plot(hcat(x̄...)', width = 2.0, label = "",
	ylims = (-2.5, 3.5))
plot(hcat(ū..., ū[end])',
    width = 2.0, linetype = :steppost)

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
visualize!(vis, model, x, Δt = h)
