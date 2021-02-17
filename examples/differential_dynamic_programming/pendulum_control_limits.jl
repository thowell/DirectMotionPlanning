include_ddp()

# Model
include_model("pendulum")
function f(model::Pendulum, x, u, w)
	mass = model.mass + w[1]
    @SVector [x[2],
              (u[1] / ((mass * model.lc * model.lc))
                - model.g * sin(x[1]) / model.lc
                - model.b * x[2] / (mass * model.lc * model.lc))]
end

n, m, d = 2, 1, 1
model = Pendulum{Midpoint, FixedTime}(n, m, d, 1.0, 0.1, 0.5, 9.81)

n = model.n
m = model.m

# Time
T = 51
h = 0.05
t = range(0, stop = h * (T - 1), length = T)

# Initial conditions, controls, disturbances
x1 = [0.0, 0.0]
xT = [π, 0.0] # goal state
ū = [1.0e-1 * rand(model.m) for t = 1:T-1]
w = [zeros(model.d) for t = 1:T-1]

# Rollout
x̄ = rollout(model, x1, ū, w, h, T)
# x̄ = linear_interpolation(x1, xT, T)
# plot(hcat(x̄...)')

# Objective
Q = [(t < T ? Diagonal(1.0 * ones(model.n))
        : Diagonal(1.0 * ones(model.n))) for t = 1:T]
q = [-2.0 * Q[t] * xT for t = 1:T]

R = [Diagonal(1.0e-1 * ones(model.m)) for t = 1:T-1]
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
p = [t < T ? 0 * 2 * m : n for t = 1:T]
ul = [-Inf]
uu = [Inf]
info_t = Dict()#:ul => ul, :uu => uu, :inequality => (1:2 * m))
info_T = Dict(:xT => xT)
con_set = [StageConstraint(p[t], t < T ? info_t : info_T) for t = 1:T]

function c!(c, cons::StageConstraints, x, u, t)
	T = cons.T
	p = cons.con[t].p

	if t < T
		# ul = cons.con[t].info[:ul]
		# uu = cons.con[t].info[:uu]
		# c .= [ul - u; u - uu]
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
    max_iter = 1000, max_al_iter = 6,
	ρ_init = 1.0, ρ_scale = 10.0)

	# verbose = true)
x, u = current_trajectory(prob)
x̄, ū = nominal_trajectory(prob)

# Visualize
using Plots
plot(t, π * ones(T),
    width = 2.0, color = :black, linestyle = :dash)
plot!(t, hcat(x...)', width = 2.0, label = "")

# plot(t, hcat([con_set[1].info[:ul] for t = 1:T]...)',
#     width = 2.0, color = :black, label = "")
# plot!(t, hcat([con_set[1].info[:uu] for t = 1:T]...)',
#     width = 2.0, color = :black, label = "")
plot(t, hcat(u..., u[end])',
    width = 2.0, linetype = :steppost,
	label = "", color = :orange)

# Simulate policy
include(joinpath(@__DIR__, "simulate.jl"))

# Model
model_sim = Pendulum{RK3, FixedTime}(n, m, d, 1.0, 0.1, 0.5, 9.81)
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
# K, _ = tvlqr(model, x̄, ū, h, Q, R)
# # K = [-k for k in K]
# K = [-K[1] for t = 1:T-1]
# plot(vcat(K...))

# Simulate
N_sim = 100
x_sim = []
u_sim = []
J_sim = []
Random.seed!(1)
for k = 1:N_sim
	wi_sim = min(0.9, max(-0.9, 1.0 * randn(1)[1]))
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
	title = "pendulum (J_avg = $(round(mean(J_sim), digits = 3)), N_sim = $N_sim)")

for xs in x_sim
	plt = plot!(t_sim, hcat(xs...)[idx, :]',
	    width = 1.0, color = :magenta, label = "")
end
display(plt)

plt = plot(t, hcat(ū..., ū[end])',
	width = 2.0, color = :black, label = "",
	xlabel = "time (s)", ylabel = "control",
	title = "pendulum (J_avg = $(round(mean(J_sim), digits = 3)), N_sim = $N_sim)")

for us in u_sim
	plt = plot!(t_sim, hcat(us..., us[end])',
		width = 1.0, color = :magenta, label = "",
		linetype = :steppost)
end
display(plt)
# u_sim
# plot(vcat(K...))
