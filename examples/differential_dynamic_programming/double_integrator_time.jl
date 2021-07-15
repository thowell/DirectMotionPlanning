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
	x + h * f(model, x + 0.5 * h * f(model, x, u, w), u, w)
end

model = DoubleIntegratorContinuous{Midpoint, FixedTime}(2, 1, 1)
n = model.n
m = model.m

# Time
T = 11
h = 0.1

# Initial conditions, controls, disturbances
x1 = [1.0; 0.0]
x_ref = [[0.0; 0.0] for t = 1:T]
xT = [0.0; 0.0]
ū = [1.0 * rand(model.m) for t = 1:T-1]
u_ref = [zeros(model.m) for t = 1:T-1]
w = [zeros(model.d) for t = 1:T-1]

# Rollout
x̄ = rollout(model, x1, ū, w, h, T)

# Objective
Q = [(t < T ? 0.0 * Diagonal([1.0; 1.0]) : 0.0 * Diagonal([1.0; 1.0])) for t = 1:T]
q = [-2.0 * Q[t] * x_ref[t] for t = 1:T]
R = [Diagonal(1.0e-5 * ones(model.m)) for t = 1:T-1]
r = [zeros(model.m) for t = 1:T-1]

obj = StageCosts([QuadraticCost(h * Q[t], h * q[t],
	t < T ? h * R[t] : nothing, t < T ? h * r[t] : nothing) for t = 1:T], T)

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
ul = [-10.0]
uu = [10.0]
p = [t < T ? 2 * m : n for t = 1:T]
info_t = Dict(:ul => ul, :uu => uu, :inequality => (1:2 * m))
info_T = Dict(:xT => xT)
con_set = [StageConstraint(p[t], t < T ? info_t : info_T) for t = 1:T]

function c!(c, cons::StageConstraints, x, u, t)
	T = cons.T
	p = cons.con[t].p

	if t < T
		ul = cons.con[t].info[:ul]
		uu = cons.con[t].info[:uu]
		c .= [ul - u; u - uu]
		nothing
	else
		c .= x - cons.con[T].info[:xT]
	end
end

prob = problem_data(model, obj, con_set, copy(x̄), copy(ū), w, h, T)

# Solve
@time constrained_ddp_solve!(prob,
	max_iter = 1000, max_al_iter = 10,
	ρ_init = 1.0, ρ_scale = 10.0,
	con_tol = 1.0e-5)

x, u = current_trajectory(prob)
x̄, ū = nominal_trajectory(prob)

objective(obj, x, u)
objective(prob.m_data.obj, x, u)

# Visualize
using Plots
t = range(0, stop = h * (T - 1), length = T)
plot(t, hcat([[0.0; 0.0] for t = 1:T]...)',
    width = 2.0, color = :black, label = "")
plot!(t, hcat([x̄[t][1:2] for t = 1:T]...)', color = :magenta, label = "")
plot(t, hcat(ū..., u[end])', linetype = :steppost)

# free final time
function f(model::DoubleIntegratorContinuous, x, u, w)
    [x[2]; u[1]]
end

function fd(model::DoubleIntegratorContinuous{Midpoint, FixedTime}, x, u, w, h, t)
	if t == 1
		# h = max(0.001, u[2])
		h = u[2] * u[2]
    	return [x + h * f(model, x + 0.5 * h * f(model, x, view(u, 1:1), w), view(u, 1:1), w); u[2]]
	else
		h = x[3] * x[3]
		return [view(x, 1:2) + h * f(model, view(x, 1:2) + 0.5 * h * f(model, view(x, 1:2), u, w), u, w); x[3]]
	end
end

model = DoubleIntegratorContinuous{Midpoint, FixedTime}(2, 1, 1)
n = [t == 1 ? model.n : model.n + 1 for t = 1:T]
m = [t == 1 ? model.m + 1 : model.m for t = 1:T]

ū_alt = [t == 1 ? [ū[t]; sqrt(h)] : ū[t] for t = 1:T-1]

# Rollout
x̄ = rollout(model, x1, ū_alt, w, nothing, T)

# Objective
obj = StageCosts([NonlinearCost() for t = 1:T], T)

function g(obj::StageCosts, x, u, t)
	T = obj.T
    if t < T
		t == 1 ? (h = u[2] * u[2]) : (h = x[3] * x[3])
        return h * (view(x, 1:2)' * Q[t] * view(x, 1:2) + q[t]' * view(x, 1:2) + view(u, 1:1)' * R[t] * view(u, 1:1) + r[t]' * view(u, 1:1)) + 1.0 * h
    elseif t == T
		h = x[3] * x[3]
        return view(x, 1:2)' * Q[t] * view(x, 1:2) + q[t]' * view(x, 1:2) + 1.0 * h
    else
        return 0.0
    end
end

# Constraints
hl = 0.1 * sqrt(h)
hu = 1.0 * sqrt(h)
p = [t == 1 ? 4 : (t == T ? 2 + 2 : 2 + 2) for t = 1:T]
info_1 = Dict(:ul => [ul[1]; hl], :uu => [uu[1]; hu], :inequality => (1:4))
info_t = Dict(:ul => ul, :uu => uu, :hl => [hl], :hu => [hu], :inequality => (1:4))
info_T = Dict(:xT => xT, :hl => [hl], :hu => [hu], :inequality => (3:4))
con_set = [StageConstraint(p[t], t < T ? (t == 1 ? info_1 : info_t) : info_T) for t = 1:T]

function c!(c, cons::StageConstraints, x, u, t)
	T = cons.T
	if t == 1
		c[1:2] .= view(u, 1:2) - cons.con[t].info[:uu]
		c[3:4] .= cons.con[t].info[:ul] - view(u, 1:2)
	end
	if t > 1 && t < T
		c[1:1] .= view(u, 1:1) - cons.con[t].info[:uu]
		c[2:2] .= cons.con[t].info[:ul] - view(u, 1:1)
		c[3:3] .= view(x, 3:3) - cons.con[t].info[:hu]
		c[4:4] .= cons.con[t].info[:hl] - view(x, 3:3)
	end
	if t == T
		c[1:2] .= view(x, 1:2) - cons.con[t].info[:xT]
		c[3:3] .= view(x, 3:3) - cons.con[t].info[:hu]
		c[4:4] .= cons.con[t].info[:hl] - view(x, 3:3)
	end
end

prob = problem_data(model, obj, con_set, copy(x̄), copy(ū_alt), w, h, T, n = n, m = m)

# Solve
@time constrained_ddp_solve!(prob,
    max_iter = 1000, max_al_iter = 10,
	ρ_init = 1.0, ρ_scale = 10.0)

# # Solve
# @time ddp_solve!(prob,
#     max_iter = 100, verbose = true)
x, u = current_trajectory(prob)
x̄, ū = nominal_trajectory(prob)
objective(obj, x, u)
objective(prob.m_data.obj, x, u)
h_opt = u[1][2] * u[1][2]
@show h_opt * (T - 1)
@show h_opt * (T - 1)

# Visualize
using Plots
t = range(0, stop = h_opt * (T - 1), length = T)
plot(t, hcat([xt[1:2] for xt in x]...)[:, 1:T]', color = :black, label = "")
plot(t, hcat([ut[1:1] for ut in u]..., u[end])', color = :black, label = "", linetype = :steppost)
