include_ddp()

# Model
include_model("acrobot")
n = model.n
m = model.m

# Time
T = 101
h = 0.1

# Initial conditions, controls, disturbances
x1 = [0.0, 0.0, 0.0, 0.0]
xT = [π, 0.0, 0.0, 0.0] # goal state
ū = [1.0e-2 * rand(model.m) for t = 1:T-1]
w = [zeros(model.d) for t = 1:T-1]
# Rollout
x̄ = rollout(model, x1, ū, w, h, T)
# x̄ = linear_interpolation(x1, xT, T)
# plot(hcat(x̄...)')

# Objective
Q = [(t < T ? Diagonal(1.0e-3 * ones(model.n))
    : Diagonal(1.0e-3 * ones(model.n))) for t = 1:T]
R = Diagonal(1.0e-3 * ones(model.m))
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

# Constraints
p = [t < T ? 2 * m : n for t = 1:T]
info_t = Dict(:ul => [-10.0], :uu => [10.0], :inequality => (1:2 * m))
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

@time x, u, obj_al = solve(model, obj, con_set, copy(x̄), copy(ū), w, h, T,
    max_iter = 1000, max_al_iter = 7,
	ρ_init = 1.0, ρ_scale = 10.0,
	con_tol = 1.0e-3,
	verbose = true)

# Visualize
using Plots
plot(π * ones(T),
    width = 2.0, color = :black, linestyle = :dash)
plot!(hcat(x...)', width = 2.0, label = "")

plot(hcat([con_set[1].info[:ul] for t = 1:T]...)',
    width = 2.0, color = :black, label = "")
plot!(hcat([con_set[1].info[:uu] for t = 1:T]...)',
    width = 2.0, color = :black, label = "")
plot!(hcat(u..., u[end])',
    width = 2.0, linetype = :steppost,
	label = "", color = :orange)

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
# open(vis)
visualize!(vis, model, x, Δt = h)
