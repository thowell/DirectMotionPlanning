using Plots
using Random
Random.seed!(1)

include_ddp()

# Model
include_model("double_integrator")

function f(model::DoubleIntegratorContinuous, x, u, w)
    [x[2]; (1.0 + w[1]) * u[1]]
end

model = DoubleIntegratorContinuous{Midpoint, FixedTime}(2, 1, 1)
n = model.n
m = model.m

struct FOHModel{I, T} <: Model{I, T}
	n::Vector{Int}
	m::Vector{Int}
	d::Vector{Int}

	model::Model
end

function foh_model(model::Model, T)
	n = [t == 1 ? model.n : model.n + model.m for t = 1:T]
	m = [t == 1 ? 2 * model.m : model.m for t = 1:T-1]
	d = [model.d for t = 1:T-1]

	FOHModel{typeof(model).parameters...}(n, m, d, model)
end

function fd(models::FOHModel, x, u, w, h, t)
	if t == 1
		return [fd(models.model,
					view(x, 1:model.n),
					view(u, 1:model.m),
					w, h, t);
				 view(u, 1:model.m) + h * view(u, model.m .+ (1:model.m))]
	else
		return [fd(models.model,
					view(x, 1:model.n),
					view(x, model.n .+ (1:model.m)),
					w, h, t);
				 view(x, model.n .+ (1:model.m)) + h * view(u, 1:model.m)]
	end
end

# Time
T = 11
h = 0.1

# FOH model
model_foh = foh_model(model, T)

# Initial conditions, controls, disturbances
x1 = [1.0; 0.0]
x_ref = [t == 1 ? zeros(model.n) : zeros(model.n + model.m) for t = 1:T]
xT = zeros(model.n + model.m)
ū = [t == 1 ? 1.0 * rand(2 * model.m) : 1.0 * rand(model.m) for t = 1:T-1]
w = [zeros(model.d) for t = 1:T-1]

# Rollout
x̄ = rollout(model_foh, x1, ū, w, h, T)

# Objective
Q = [(t > 1 ?
	 Diagonal([1.0; 1.0; 1.0e-3])
		: Diagonal([1.0; 1.0])) for t = 1:T]
q = [-2.0 * Q[t] * x_ref[t] for t = 1:T]
R = [t == 1 ? Diagonal(1.0e-3 * ones(2 * model.m)) : Diagonal(1.0e-3 * ones(model.m)) for t = 1:T-1]
r = [t == 1 ? zeros(2 * model.m) : zeros(model.m) for t = 1:T-1]

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
# ul = [-10.0]
# uu = [10.0]
p = [t < T ? 0 : model.n for t = 1:T]
info_t = Dict()#:ul => ul, :uu => uu, :inequality => (1:2 * m))
info_T = Dict(:xT => xT[1:model.n])
con_set = [StageConstraint(p[t], t < T ? info_t : info_T) for t = 1:T]

function c!(c, cons::StageConstraints, x, u, t)
	T = cons.T
	p = cons.con[t].p

	if t < T
		# ul = cons.con[t].info[:ul]
		# uu = cons.con[t].info[:uu]
		# c .= [ul - u; u - uu]
	else
		c .= view(x, 1:model.n) - cons.con[T].info[:xT]
	end
end

prob = problem_data(model_foh, obj, con_set, copy(x̄), copy(ū), w, h, T,
	n = model_foh.n, m = model_foh.m, d = model_foh.d)

# Solve
@time constrained_ddp_solve!(prob,
    max_iter = 100, verbose = true)

x, u = current_trajectory(prob)
x̄, ū = nominal_trajectory(prob)
hcat([x[1:model.n] for t = 1:T]...)

# Visualize
using Plots
plot(hcat([xT[1:model.n] for t = 1:T]...)',
    width = 2.0, color = :black, label = "")
plot!(hcat([x[t][1:model.n] for t = 1:T]...)', color = :magenta, label = "")

u_foh = [u[1][1:model.m], [x[t][model.n .+ (1:model.m)] for t = 2:T]...]
plot(hcat(u_foh...)', linetype = :steppost)

plot(hcat(u..., u[end])', linetype = :steppost)
# Simulate policy
include(joinpath(@__DIR__, "simulate.jl"))

# Model
model_sim = DoubleIntegratorContinuous{RK3, FixedTime}(model.n, model.m, model.d)
x1_sim = copy(x1)
T_sim = 10 * T

# Time
tf = h * (T - 1)
t = range(0, stop = tf, length = T)
t_sim = range(0, stop = tf, length = T_sim)
dt_sim = tf / (T_sim - 1)

# Policy
# K = [K for K in prob.p_data.K]
# plot(vcat(K...))
# # K = [prob.p_data.K[t] for t = 1:T-1]
# K, _ = tvlqr(model, x̄, ū, h, Q, R)
# # K = [-k for k in K]
# K = [-K[t] for t = 1:T-1]
# plot(vcat(K...))

# Simulate
N_sim = 100
x_sim = []
u_sim = []
J_sim = []
Random.seed!(1)
for k = 1:N_sim
	wi_sim = min(0.1, max(-0.1, 1.0e-1 * randn(1)[1]))
	w_sim = [wi_sim for t = 1:T-1]
	println("sim: $k - w = $(wi_sim[1])")

	x_ddp, u_ddp, J_ddp, Jx_ddp, Ju_ddp = simulate_linear_feedback(
		model_sim,
		K,
	    x̄, ū,
		x_ref, u_ref,
		Q, R,
		T_sim, h,
		x1_sim,
		w_sim,
		ul = ul,
		uu = uu)

	push!(x_sim, x_ddp)
	push!(u_sim, u_ddp)
	push!(J_sim, J_ddp)
end

# Visualize
idx = (1:2)
plt = plot(t, hcat(x̄...)[idx, :]',
	width = 2.0, color = :black, label = "",
	xlabel = "time (s)", ylabel = "state",
	title = "double integrator (J_avg = $(round(mean(J_sim), digits = 3)), N_sim = $N_sim)")

for xs in x_sim
	plt = plot!(t_sim, hcat(xs...)[idx, :]',
	    width = 1.0, color = :magenta, label = "")
end
display(plt)

plt = plot(t, hcat(ū..., ū[end])',
	width = 2.0, color = :black, label = "",
	xlabel = "time (s)", ylabel = "control",
	title = "double integrator (J_avg = $(round(mean(J_sim), digits = 3)), N_sim = $N_sim)")

for us in u_sim
	plt = plot!(t_sim, hcat(us..., us[end])',
		width = 1.0, color = :magenta, label = "",
		linetype = :steppost)
end
display(plt)
# u_sim
# plot(vcat(K...))
