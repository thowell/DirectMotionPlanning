using Plots
using Random
Random.seed!(1)

include_ddp()

# Model
include_model("double_integrator")

function f(model::DoubleIntegratorContinuous, x, u, w)
    [x[2]; u[1]]
end

function fd(model::DoubleIntegratorContinuous{Midpoint, FixedTime}, x, u, w, h, t)
	if t == 1
		x1 = u[2:3]
    	return x1 + h * f(model, x1 + 0.5 * h * f(model, x1, u, w), u, w)
	else
		return x + h * f(model, x + 0.5 * h * f(model, x, u, w), u, w)
	end
end

# Time
T = 11
h = 0.1

model = DoubleIntegratorContinuous{Midpoint, FixedTime}(2, 1, 0)
n = [model.n for t = 1:T]
m = [t == 1 ? model.m + model.n : model.m for t = 1:T]

# Initial conditions, controls, disturbances
x1 = [0.0; 0.0]
xT = [2.0; 0.0]
ū = [0.01 * randn(m[t]) for t = 1:T-1]
w = [zeros(model.d) for t = 1:T-1]

# Rollout
x̄ = rollout(model, x1, ū, w, h, T)

# Objective
Q = [(t < T ? h : 1.0) * 0.0 * Diagonal([1.0; 0.1]) for t = 1:T]
q = [-2.0 * Q[t] * xT for t = 1:T]
R = [h * Diagonal(1.0 * ones(m[t])) for t = 1:T-1]
r = [zeros(m[t]) for t = 1:T-1]
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
p = [t == 1 ? 2 * n[1] : (t == T ? n[T] : 0) for t = 1:T]
info_1 = Dict(:x1l => [-1.0; -0.1], :x1u => [1.0; 0.1], :inequality => (1:2 * n[1]))
info_t = Dict()
info_T = Dict(:xT => xT)
con_set = [StageConstraint(p[t], t < T ? (t == 1 ? info_1 : info_t) : info_T) for t = 1:T]

function c!(c, cons::StageConstraints, x, u, t)
	T = cons.T
	if t == 1
		c[1:2] .= view(u, 2:3) - cons.con[t].info[:x1u]
		c[3:4] = cons.con[t].info[:x1l] - view(u, 2:3)
	end
	if t == T
		c[1:2] .= view(x, 1:2) - cons.con[t].info[:xT]
	end
end

prob = problem_data(model, obj, con_set, copy(x̄), copy(ū), w, h, T, n = n, m = m)

# Solve
@time constrained_ddp_solve!(prob,
    max_iter = 1000, max_al_iter = 6,
	ρ_init = 1.0, ρ_scale = 10.0)

# # Solve
# @time ddp_solve!(prob,
#     max_iter = 100, verbose = true)

x, u = current_trajectory(prob)
x̄, ū = nominal_trajectory(prob)

x[1] = u[1][2:3]
x̄[1] = ū[1][2:3]

# Visualize
using Plots
plot(hcat(x...)[:, 1:T]', color = :black, label = "")
# plot(hcat(u..., u[end])', linetype = :steppost)
