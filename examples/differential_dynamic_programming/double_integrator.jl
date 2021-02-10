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

# Time
T = 101
h = 0.01

z = range(0.0, stop = 2.0 * 2.0 * π, length = T)
p_ref = zeros(T) #p_ref = 1.0 * cos.(2.0 * z)
plot(z, p_ref)

# Initial conditions, controls, disturbances
x1 = [1.0; 0.0]
x_ref = [[p_ref[t]; 0.0] for t = 1:T]
ū = [0.1 * rand(model.m) for t = 1:T-1]
w = [zeros(model.d) for t = 1:T-1]

# Rollout
x̄ = rollout(model, x1, ū, w, h, T)

# Objective
Q = [(t < T ?
	 h * Diagonal([1.0; 1.0])
		: Diagonal([1000.0; 1000.0])) for t = 1:T]
q = [-2.0 * Q[t] * x_ref[t] for t = 1:T]
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
ul = [-5.0]
uu = [5.0]
p = [t < T ? 2 * m : 0 for t = 1:T]
info_t = Dict(:ul => ul, :uu => uu, :inequality => (1:2 * m))
info_T = Dict()
con_set = [StageConstraint(p[t], t < T ? info_t : info_T) for t = 1:T]

function c!(c, cons::StageConstraints, x, u, t)
	T = cons.T
	p = cons.con[t].p

	if t < T
		ul = cons.con[t].info[:ul]
		uu = cons.con[t].info[:uu]
		c .= [ul - u; u - uu]
	else
		nothing
	end
end

prob = problem_data(model, obj, con_set, copy(x̄), copy(ū), w, h, T)

# Solve
@time constrained_ddp_solve!(prob,
    max_iter = 100, verbose = true)

x, u = current_trajectory(prob)
x̄, ū = nominal_trajectory(prob)

# Visualize
using Plots
plot(hcat([[p_ref[t]; 0.0] for t = 1:T]...)',
    width = 2.0, color = :black, label = "")
plot!(hcat(x...)', color = :magenta, label = "")
plot(hcat(u..., u[end])', linetype = :steppost)

# Simulate policy
include(joinpath(@__DIR__, "simulate.jl"))

# Model
model_sim = model
x1_sim = copy(x1)
T_sim = 10 * T

# Time
tf = h * (T - 1)
t = range(0, stop = tf, length = T)
t_sim = range(0, stop = tf, length = T_sim)
dt_sim = tf / (T_sim - 1)

# Policy
K = [K for K in prob.p_data.K]
plot(vcat(K...))
K = [prob.p_data.K[t] for t = 1:T-1]
K, _ = tvlqr(model, x̄, ū, h, Q, R)
# K = [-k for k in K]
K = [-K[1] for t = 1:T-1]
plot(vcat(K...))

# Simulate
N_sim = 100
x_sim = []
u_sim = []
J_sim = []
Random.seed!(1)
for k = 1:N_sim
	wi_sim = 1.0e-1 * randn(1)
	w_sim = [wi_sim for t = 1:T-1]
	println("sim: $k - w = $(wi_sim[1])")

	x_ddp, u_ddp, J_ddp, Jx_ddp, Ju_ddp = simulate_linear_feedback(
		model_sim,
		K,
	    x̄, ū,
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
