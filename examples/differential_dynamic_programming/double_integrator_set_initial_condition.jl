using Plots
using Random
Random.seed!(1)

include_ddp()

# Model
include_model("double_integrator")

function f(model::DoubleIntegratorContinuous, x, u, w)
    [x[2] + w[1]; (1.0 + w[3]) * u[1] + w[2]]
end

function fd(model::DoubleIntegratorContinuous{Midpoint, FixedTime}, x, u, w, h, t)
	if t == 1
		x1 = u[2:3]
    	return x1 + h * f(model, x1 + 0.5 * h * f(model, x1, u, w), u, w)
	else
		return x + h * f(model, x + 0.5 * h * f(model, x, u, w), u, w)
	end
end

model = DoubleIntegratorContinuous{Midpoint, FixedTime}(2, 3, 3)
n = model.n
m = model.m

# Time
T = 101
h = 0.1

z = range(0.0, stop = 3.0 * 2.0 * π, length = T)
p_ref = 1.0 * cos.(1.0 * z)
plot(z, p_ref)

# Initial conditions, controls, disturbances
x1 = [p_ref[1]; 0.0]
x1_alt = [0.0; 0.0]
ū = [rand(model.m) for t = 1:T-1]
w = [zeros(model.d) for t = 1:T-1]

# Rollout
x̄ = rollout(model, x1, ū, w, h, T)

# Objective
Q = Diagonal([100.0; 0.1])
R = Diagonal(0.01 * ones(model.m))

obj = StageCosts([QuadraticCost(Q, nothing,
	t < T ? R : nothing, nothing) for t = 1:T], T)

function g(obj::StageCosts, x, u, t)
	T = obj.T
    if t < T
		Q = obj.cost[t].Q
	    R = obj.cost[t].R
        return h * (x - [p_ref[t]; 0.0])' * Q * (x - [p_ref[t]; 0.0]) + u' * R * u
    elseif t == T
		Q = obj.cost[T].Q
        return (x - [p_ref[T]; 0.0])' * Q * (x - [p_ref[T]; 0.0])
    else
        return 0.0
    end
end

# g(obj, x̄[T], nothing, T)
# objective(obj, x̄, ū)

# Constraints
p = [t < T ? n : 0 for t = 1:T]
info_t = Dict()#:ul => [-5.0], :uu => [5.0], :inequality => (1:2 * m))
info_T = Dict()#:xT => xT)
con_set = [StageConstraint(p[t], t < T ? info_t : info_T) for t = 1:T]

function c!(c, cons::StageConstraints, x, u, t)
	T = cons.T
	p = cons.con[t].p

	if t < T
		if t == 1
			c .= u[2:3] - x1_alt
		else
			c .= u[2:3]
		end
	else
		nothing
	end
end

prob = problem_data(model, obj, con_set, copy(x̄), copy(ū), w, h, T)

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
plot(hcat([[p_ref[t]; 0.0] for t = 1:T]...)',
    width = 2.0, color = :black, label = "")
plot!(hcat(x...)', color = :magenta, label = "")
# plot(hcat(u..., u[end])', linetype = :steppost)
