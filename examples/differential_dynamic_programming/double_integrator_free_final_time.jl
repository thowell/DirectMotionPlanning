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
		# h = max(0.001, u[2])
		h = u[2]
    	return [x + h * f(model, x + 0.5 * h * f(model, x, view(u, 1:1), w), view(u, 1:1), w); h]
	else
		h = x[3]
		return [view(x, 1:2) + h * f(model, view(x, 1:2) + 0.5 * h * f(model, view(x, 1:2), u, w), u, w); h]
	end
end

# Time
T = 11
h = 0.1

model = DoubleIntegratorContinuous{Midpoint, FixedTime}(2, 1, 1)
n = [t == 1 ? model.n : model.n + 1 for t = 1:T]
m = [t == 1 ? model.m + 1 : model.m for t = 1:T]

# Initial conditions, controls, disturbances
x1 = [0.0; 0.0]
xT = [1.0; 0.0]
ū = [t == 1 ? [0.01 * randn(1); h] : 0.01 * randn(1) for t = 1:T-1]
w = [zeros(model.d) for t = 1:T-1]

# Rollout
x̄ = rollout(model, x1, ū, w, h, T)

# Objective
obj = StageCosts([NonlinearCost() for t = 1:T], T)

Q = [t < T ? Diagonal(1.0e-6 * [1.0; 0.1]) : Diagonal(1.0e-6 * [1.0; 0.1]) for t = 1:T]
q = [-2.0 * Q[t] * xT for t = 1:T]
R = [Diagonal(1.0e-6 * ones(1)) for t = 1:T-1]
r = [[0.0] for t = 1:T-1]

function g(obj::StageCosts, x, u, t)
	T = obj.T
    if t < T
		t == 1 ? (h = u[2]) : (h = x[3])
        return 1.0 * h^2.0 + 1.0e-5 * u[1]^2.0 #h * (view(x, 1:2)' * Q[t] * view(x, 1:2) + q[t]' * view(x, 1:2) + view(u, 1:1)' * R[t] * view(u, 1:1) + r[t]' * view(u, 1:1)) + 1.0 * h^2.0
    elseif t == T
        return 0.0 # view(x, 1:2)' * Q[t] * view(x, 1:2) + q[t]' * view(x, 1:2)
    else
        return 0.0
    end
end

# Constraints
p = [t == 1 ? 4 : (t == T ? 2 : 2) for t = 1:T]
info_1 = Dict(:ul => [-5.0; 0.001], :uu => [5.0; 1.0], :inequality => (1:4))
info_t = Dict(:ul => [-5.0], :uu => [5.0], :inequality => (1:2))
info_T = Dict(:xT => xT)
con_set = [StageConstraint(p[t], t < T ? (t == 1 ? info_1 : info_t) : info_T) for t = 1:T]

function c!(c, cons::StageConstraints, x, u, t)
	T = cons.T
	if t == 1
		c[1:2] .= view(u, 1:2) - cons.con[t].info[:uu]
		c[3:4] = cons.con[t].info[:ul] - view(u, 1:2)
	end
	if t > 1 && t < T
		c[1:1] .= view(u, 1:1) - cons.con[t].info[:uu]
		c[2:2] = cons.con[t].info[:ul] - view(u, 1:1)
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

@show u[1][2] * (T - 1)
@show ū[1][2] * (T - 1)

# Visualize
using Plots
plot(hcat([xt[1:2] for xt in x]...)[:, 1:T]', color = :black, label = "")
plot(hcat([ut[1:1] for ut in u]...)[:, 1:T-1]', color = :black, label = "", linetype = :steppost)
