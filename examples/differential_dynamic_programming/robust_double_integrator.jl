using Plots
using Random
Random.seed!(1)

# Differential Dynamic Programming
include_ddp()
include(joinpath(pwd(), "src/differential_dynamic_programming/multiple_models.jl"))

# Model
include_model("double_integrator")

function f(model::DoubleIntegratorContinuous, x, u, w)
    [x[2] + w[1]; (1.0 + w[3]) * u[1] + w[2]]
end

model = DoubleIntegratorContinuous{Midpoint, FixedTime}(2, 1, 3)
n = model.n
m = model.m

# Models
N = 1000#2 * n + 1

# Time
T = 101
h = 0.01
tf = h * (T - 1)
t = range(0, stop = tf, length = T)

# Reference position trajectory
# z = range(0.0, stop = 3.0 * 2.0 * π, length = T)
# p_ref = 1.0 * cos.(1.0 * z)
# plot(z, p_ref)
p_ref = [0.0 for t = 1:T]

# Initial conditions, controls, disturbances
# x1 = [p_ref[1]; 0.0]
x1 = [10.0; 0.0]
ū = [[0.1 * rand(model.m) for t = 1:T-1] for i = 1:N]

W = Distributions.MvNormal(zeros(model.d), Diagonal([0.0, 0.0, 10.0]))
# w = [[zeros(model.d) for t = 1:T-1], [[rand(W, 1) for t = 1:T-1] for i = 1:N-1]...]
# w = [[vec(rand(W, 1)) for t = 1:T-1] for i = 1:N]
w = [[zeros(model.d) for t = 1:T-1] for i = 1:N]

# Rollout
x̄ = [rollout(model, x1, ū[i], w[i], h, T) for i = 1:N]

# x_ref = [sum([x̄i[t] for x̄i in x̄]) ./ N for t = 1:T]
# u_ref = [sum([ūi[t] for ūi in ū]) ./ N for t = 1:T-1]

p_idx = (1:1)
plt = plot()
for i = 1:N
	plt = plot!(t, hcat(x̄[i]...)[p_idx, :]',
		label = "")
end
display(plt)

# Objective
Q = Diagonal([100.0; 1.0])
R = Diagonal(0.01 * ones(model.m))

obj = StageCosts([QuadraticCost(Q, nothing,
	t < T ? R : nothing, nothing) for t = 1:T], T)

function g(obj::StageCosts, x, u, t)
	T = obj.T
    if t < T
		Q = obj.cost[t].Q
	    R = obj.cost[t].R
        return h * ((x - [p_ref[t]; 0.0])' * Q * (x - [p_ref[t]; 0.0]) + u' * R * u)
    elseif t == T
		Q = obj.cost[T].Q
        return (x - [p_ref[T]; 0.0])' * Q * (x - [p_ref[T]; 0.0])
    else
        return 0.0
    end
end

# g(obj, x̄[1][T], nothing, T)
# objective(obj, x̄[1], ū[1])

# Model(s) data



models_data = models(model, obj, deepcopy(x̄), deepcopy(ū), w, h, T)
# objective(obj, x̄[4], ū[4])
#
#
# # allocate policy data
# p_data = policy_data(model, T)
#
# # allocate solver data
# s_data = solver_data(model, T)
#
# # compute objective
#
#
# J = objective(models_data, mode = :nominal)
#
#
#
# derivatives!(models_data)
#
#
#
# backward_pass!(p_data, models_data)
#
#
#
# s_data.obj = J
#
# forward_pass!(p_data, models_data, s_data)
# s_data.obj

function ddp_solve!(prob::ProblemData;
    max_iter = 10,
    grad_tol = 1.0e-5,
    verbose = true)

	println()
    verbose && println("Differential Dynamic Programming")

	# data
	p_data = prob.p_data
	m_data = prob.m_data
	s_data = prob.s_data

    # compute objective
    s_data.obj = objective(m_data, mode = :nominal)

	N = length(prob.m_data)

    for i = 1:max_iter
        # derivatives
        derivatives!(m_data)

        # backward pass
        backward_pass!(p_data, m_data)

		# # set nominal trajectories
		# x_ref = [sum([m.x̄[t] for m in prob.m_data]) ./ N for t = 1:T]
		# u_ref = [sum([m.ū[t] for m in prob.m_data]) ./ N for t = 1:T-1]
		#
		# # disturbances
		# w = [[rand(W, 1) for t = 1:T-1] for i = 1:N]
		#
		# for i = 1:N
		# 	# m_data[i].x̄ .= deepcopy(x_ref)
		# 	# m_data[i].ū .= deepcopy(u_ref)
		# 	m_data[i].w .= deepcopy(w[i])
		# end

        # forward pass
        forward_pass!(p_data, m_data, s_data)

        # check convergence
        grad_norm = norm(s_data.gradient, Inf)
        verbose && println("     iter: $i
             cost: $(s_data.obj)
			 grad_norm: $(grad_norm)")
		grad_norm < grad_tol && break
        !s_data.status && break
    end
end

prob = problem_data(models_data)

# w = [[rand(W, 1) for t = 1:T-1] for i = 1:N]
# for i = 1:N
	# m_data[i].x̄ .= deepcopy(x_ref)
	# m_data[i].ū .= deepcopy(u_ref)
	# prob.m_data[i].w .= deepcopy(w[i])
	# rollout!(prob.p_data, prob.m_data[i], α = 1.0)
# end
prob.p_data
prob.m_data[1].obj_deriv.gx
# Solve
@time ddp_solve!(prob,
    max_iter = 1000, verbose = true,
	grad_tol = 1.0e-8)

x = [m.x for m in prob.m_data]
u = [m.u for m in prob.m_data]

x̄ = [m.x̄ for m in prob.m_data]
ū = [m.ū for m in prob.m_data]

x_ref = [sum([m.x̄[t] for m in prob.m_data]) ./ N for t = 1:T]
u_ref = [sum([m.ū[t] for m in prob.m_data]) ./ N for t = 1:T-1]

# Visualize
idx = (1:1)
_plt = plot(hcat([[p_ref[t]; 0.0][idx] for t = 1:T]...)',
    width = 2.0, color = :black, label = "")
for i = 1:N
	_plt = plot!(hcat(x[i]...)[idx, :]', color = :magenta, label = "")
end
_plt = plot!(hcat(x_ref...)[idx, :]', color = :orange, width = 5.0, label = "")
display(_plt)
# plot(hcat(u..., u[end])', linetype = :steppost)

# Simulate policy
include_dpo()
include(joinpath(pwd(), "examples/direct_policy_optimization/simulate.jl"))

# Model
model_sim = model
x1_sim = copy(x1)
T_sim = 10 * (T-1) + 1

# Disturbance distributions
W_sim = W#Distributions.MvNormal(zeros(model.d), Diagonal([0.0, 0.0, 1.0e-1]))
w_sim = rand(W_sim, T_sim)

# Time
t_sim = range(0, stop = tf, length = T_sim)
dt_sim = tf / (T_sim - 1)

# Policy
policy = linear_feedback(model.n, model.m)
# K, P = tvlqr(model, x̄, ū, h, [Q for t = 1:T], [R for t = 1:T-1])
K = [-K for K in prob.p_data.K]

# Simulate
N_sim = 100
x_sim = []
u_sim = []
J_sim = []
for i = 1:N_sim
	println("sim: $i")
	w_sim = rand(W_sim, T_sim)

	x_ddp, u_ddp, J_ddp, Jx_ddp, Ju_ddp = simulate(
		model_sim,
		linear_feedback(model.n, model.m), K,
	    x_ref, u_ref,
		[Q * (t < T ? h : 1.0) for t = 1:T], [R for t = 1:T-1],
		T_sim, h,
		x1_sim,
		w_sim,
		ul = [-1000.0],
		uu = [500.0])
	push!(x_sim, x_ddp)
	push!(u_sim, u_ddp)
	push!(J_sim, J_ddp)
end

# Visualize
idx = (1:2)
_plt = plot(t, hcat(x_ref...)[idx, :]',
    width = 2.0, color = :black, label = "")

for (i, xi) in enumerate(x_sim)
	_plt = plot!(t_sim, hcat(xi...)[idx, :]',
    	width = 1.0, color = :magenta, label = i == 1 ? "sim" : "")
end
_plt = plot!(t, hcat(x_ref...)[idx, :]',
    width = 2.0, color = :black, label = "ref")
_plt = plot!(; xlabel = "time (s)",
	ylabel = "position",
	title = "N_mc = $N, N_sim = $(N_sim), J_avg = $(round(mean(J_sim), digits = 2))")
display(_plt)

_plt = plot(t, hcat(u_ref..., u_ref[end])[1:1, :]',
    width = 2.0, color = :black, label = "")
for (i, ui) in enumerate(u_sim)
	_plt = plot!(t_sim, hcat(ui..., ui[end])[1:1, :]',
    	width = 1.0, color = :magenta, label = i == 1 ? "sim" : "")
end
_plt = plot!(t, hcat(u_ref..., u_ref[end])[1:1, :]',
    width = 2.0, color = :black, label = "ref")
_plt = plot!(; xlabel = "time (s)",
	ylabel = "control",
	title = "N_mc = $N, N_sim = $(N_sim), J_avg = $(round(mean(J_sim), digits = 2))")
display(_plt)
@show mean(J_sim)
round(mean(J_sim), digits = 2)
