using Plots
using Random
Random.seed!(1)

# Differential Dynamic Programming
include_ddp()
include(joinpath(pwd(), "src/differential_dynamic_programming/multiple_models.jl"))

include_model("acrobot")

function M(model::Acrobot, x, w)
    a = (model.J1 + model.J2 + model.m2 * model.l1 * model.l1
         + 2.0 * model.m2 * model.l1 * model.lc2 * cos(x[2]))

    b = model.J2 + model.m2 * model.l1 * model.lc2 * cos(x[2])

    c = model.J2

    @SMatrix [a b;
              b c]
end

function τ(model::Acrobot, x, w)
    a = (-1.0 * model.m1 * model.g * model.lc1 * sin(x[1])
         - model.m2 * model.g * (model.l1 * sin(x[1])
         + model.lc2 * sin(x[1] + x[2])))

    b = -1.0 * model.m2 * model.g * model.lc2 * sin(x[1] + x[2])

    @SVector [a,
              b]
end

function C(model::Acrobot, x, w)
    a = -2.0 * model.m2 * model.l1 * model.lc2 * sin(x[2]) * x[4]
    b = -1.0 * model.m2 * model.l1 * model.lc2 * sin(x[2]) * x[4]
    c = model.m2 * model.l1 * model.lc2 * sin(x[2]) * x[3]
    d = 0.0

    @SMatrix [a b;
              c d]
end

function B(model::Acrobot, x, w)
    @SMatrix [0.0;
              1.0]
end

function f(model::Acrobot, x, u, w)
    q = view(x, 1:2)
    v = view(x, 3:4)
    qdd = M(model, q, w) \ (-1.0 * C(model, x, w) * v
            + τ(model, q, w) + B(model, q, w) * u[1:1] - [model.b1 + w[1]; model.b2 + w[2]] .* v)
    @SVector [x[3],
              x[4],
              qdd[1],
              qdd[2]]
end
state_output(model::Acrobot, x) = x

n, m, d = 4, 1, 2
model = Acrobot{Midpoint, FixedTime}(n, m, d, 1.0, 0.33, 1.0, 0.5, 1.0, 0.33, 1.0, 0.5, 9.81, 0.1, 0.1)

n = model.n
m = model.m
d = model.d

# Models
N = 100#2 * n + 1

# Time
T = 101
h = 0.05
tf = h * (T - 1)
t = range(0, stop = tf, length = T)

# Initial conditions, controls, disturbances
# x1 = [p_ref[1]; 0.0]
x1 = [0.0, 0.0, 0.0, 0.0]
xT = [π, 0.0, 0.0, 0.0] # goal state
_ū = [1.0e-1 * rand(model.m) for t = 1:T-1]
_ū = u_ref
ū = [_ū for i = 1:N]

W = Distributions.MvNormal(zeros(model.d), Diagonal([1.0e-8, 1.0e-8]))
wi = [vec(rand(W, 1)) for i = 1:N]
w = [[zeros(model.d) for t = 1:T-1], [[wi[i] for t = 1:T-1] for i = 1:N-1]...]
# w = [[vec(rand(W, 1)) for t = 1:T-1] for i = 1:N]
# w = [[zeros(model.d) for t = 1:T-1] for i = 1:N]

# Rollout
x̄ = [rollout(model, x1, ū[1], w[i], h, T) for i = 1:N]

p_idx = (1:2)
plt = plot()
for i = 1:N
	plt = plot!(t, hcat(x̄[i]...)[p_idx, :]',
		label = "")
end
display(plt)

# Objective
Q = [(t < T ? Diagonal(1.0e-3 * ones(model.n))
    : Diagonal(100.0 * ones(model.n))) for t = 1:T]
R = [Diagonal(1.0e-5 * ones(model.m)) for t = 1:T-1]
obj = StageCosts([QuadraticCost(Q[t], nothing,
	t < T ? R[t] : nothing, nothing) for t = 1:T], T)

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


models_data = models(model, obj, deepcopy(x̄), deepcopy(ū), w, h, T)

prob = problem_data(models_data)

function ddp_solve!(prob::ProblemData;
    max_iter = 100,
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

		if i > 1
			for i = 2:N
				m_data[i].x̄ .= deepcopy(m_data[1].x̄)
				m_data[i].ū .= deepcopy(m_data[1].ū)
				# m_data[i].w .= deepcopy(w[i])
			end
		end
        # forward pass
        forward_pass!(p_data, m_data, s_data)

		# if i > 1
		# 	for i = 2:N
		# 		m_data[i].x̄ .= deepcopy(m_data[1].x̄)
		# 		m_data[i].ū .= deepcopy(m_data[1].ū)
		# 		# m_data[i].w .= deepcopy(w[i])
		# 	end
		# end
		# # set nominal trajectories
		# x_ref = [sum([m.x̄[t] for m in prob.m_data]) ./ N for t = 1:T]
		# u_ref = [sum([m.ū[t] for m in prob.m_data]) ./ N for t = 1:T-1]

        # check convergence
        grad_norm = norm(s_data.gradient, Inf)
        verbose && println("     iter: $i
             cost: $(s_data.obj)
			 grad_norm: $(grad_norm)")
		grad_norm < grad_tol && break
        !s_data.status && break
    end
end

objective(prob.m_data)
prob.m_data[1].x̄

# Solve
@time ddp_solve!(prob,
    max_iter = 250, verbose = true,
	grad_tol = 1.0e-8)

x = [m.x for m in prob.m_data]
u = [m.u for m in prob.m_data]

x̄ = [m.x̄ for m in prob.m_data]
ū = [m.ū for m in prob.m_data]

x_ref = deepcopy(prob.m_data[1].x̄)#[sum([m.x̄[t] for m in prob.m_data]) ./ N for t = 1:T]
u_ref = deepcopy(prob.m_data[1].ū)#[sum([m.ū[t] for m in prob.m_data]) ./ N for t = 1:T-1]

# Visualize
idx = (1:2)
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
Random.seed!(1)
W_sim = W#Distributions.MvNormal(zeros(model.d), Diagonal([0.0, 0.0, 1.0e-1]))
w_sim = rand(W_sim, T_sim)
for i = 1:N_sim
	println("sim: $i")
	wi = vec(rand(W_sim, 1))
	w_sim = hcat([wi for t = 1:T_sim]...)

	x_ddp, u_ddp, J_ddp, Jx_ddp, Ju_ddp = simulate(
		model_sim,
		linear_feedback(model.n, model.m), K,
	    x_ref, u_ref,
		Q, R,
		T_sim, h,
		x1_sim,
		w_sim)

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
_plt = plot!(ylims = (-2, π + 2.0); xlabel = "time (s)",
	ylabel = "state",
	title = "N_mc = $N, N_sim = $(N_sim), J_avg = $(round(mean(J_sim), digits = 3))",
	legend = :bottomright)
display(_plt)

# _plt = plot(t, hcat(u_ref..., u_ref[end])[1:1, :]',
#     width = 2.0, color = :black, label = "", linetype = :steppost)
# for (i, ui) in enumerate(u_sim)
# 	_plt = plot!(t_sim, hcat(ui..., ui[end])[1:1, :]',
#     	width = 1.0, color = :magenta, label = i == 1 ? "sim" : "")
# end
# _plt = plot!(t, hcat(u_ref..., u_ref[end])[1:1, :]',
#     width = 2.0, color = :black, label = "ref", linetype = :steppost)
# _plt = plot!(ylims = (-20.0, 20.0); xlabel = "time (s)",
# 	ylabel = "control",
# 	title = "N_mc = $N, N_sim = $(N_sim), J_avg = $(round(mean(J_sim), digits = 2))",
# 	linetype = :steppost)
# display(_plt)
@show mean(J_sim) # 0.00157
round(mean(J_sim), digits = 3)
