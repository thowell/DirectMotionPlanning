using Plots
using Random
Random.seed!(1)

# soft rect
soft_rect(z) = log(1.0 + exp(z))
zz = range(-5.0, stop = 5.0, length = 100)
plot(zz, soft_rect.(zz))
plot(zz, 5.0 * tanh.(zz))

# ddp
include_ddp()

# Model
include_model("double_integrator")

model = DoubleIntegratorContinuous{RK3, FixedTime}(2, 1, 1)
n = model.n
m = model.m

struct MultipleModel{I, T} <: Model{I, T}
	n::Vector{Int}
	m::Vector{Int}
	d::Vector{Int}

	model::Model

	N::Int
	p::Int
end

function multiple_model(model, T, N; p = 0)
	n = model.n
	m = model.m
	d = model.d

	n = [N * n + (t != 1 ? p + N * m : 0) for t = 1:T]
	m = [N * m + (t == 1 ? p : 0) for t = 1:T-1]
	d = [N * d for t = 1:T-1]

	MultipleModel{typeof(model).parameters...}(n, m, d, model, N, p)
end

# Policy
dp = 1
p_policy = dp * model.n * model.n + dp * model.n + dp * model.n * dp * model.n + dp * model.n + model.m * dp * model.n + model.m
# p_policy = dp * model.n * model.n + dp * model.n + model.m * dp * model.n + model.m
# p_policy = model.m * model.n + model.m

function policy(θ, x, t, n, m)
	# 2 layer
	p_policy = dp * model.n * model.n + dp * model.n + dp * model.n * dp * model.n + dp * model.n + model.m * dp * model.n + model.m

	K1 = reshape(view(θ, 1:(dp * n) * n), dp * n, n)
	k1 = view(θ, (dp * n) * n .+ (1:(dp * n)))

	K2 = reshape(view(θ, dp * n * n + dp * n .+ (1:(dp * n * dp * n))), dp * n, dp * n)
	k2 = view(θ, dp * n * n + dp * n + dp * n * dp * n .+ (1:(dp * n)))

	Ko = reshape(view(θ, dp * n * n + dp * n + dp * n * dp * n + dp * n .+ (1:(m * dp * n))), m, dp * n)
	ko = view(θ, dp * n * n + dp * n + dp * n * dp * n + dp * n + m * dp * n .+ (1:m))

	# z1 = soft_rect.(K1 * x + k1)
	# z2 = soft_rect.(K2 * z1 + k2)
	z1 = tanh.(K1 * x + k1)
	z2 = tanh.(K2 * z1 + k2)
	zo = Ko * z2 + ko

	## 1 layer
	# p_policy = dp * model.n * model.n + dp * model.n + model.m * dp * model.n + model.m
	#
	# K1 = reshape(view(θ, 1:(dp * n) * n), dp * n, n)
	# k1 = view(θ, (dp * n) * n .+ (1:(dp * n)))
	#
	# z1 = soft_rect.(K1 * x + k1)
	# Ko = reshape(view(θ, dp * n * n + dp * n .+ (1:m * (dp * n))), m, dp * n)
	# ko = view(θ, dp * n * n + dp * n + m * dp * n .+ (1:m))
	#
	# zo = Ko * z1 + ko

	## affine
	# p_policy = model.m * model.n + model.m
	#
	# Ko = reshape(view(θ, 1:m * n), m, n)
	# ko = view(θ, m * n .+ (1:m))
	#
	# zo = Ko * x + ko



	return zo
end

function f(model::DoubleIntegratorContinuous, x, u, w)
    [x[2];
	 (1.0 + w[1]) * (u[1]
	 	+ policy(view(u, 1 .+ (1:p_policy)), x, nothing, model.n, model.m)[1])]
end

function fd(models::MultipleModel, x, u, w, h, t)
	N = models.N

	n = models.n[t]
	m = models.m[t]
	d = models.d[t]

	ni = models.model.n
	mi = models.model.m
	di = models.model.d

	p = models.p

	x⁺ = []
	u_policy = []

	if t == 1
		θ = view(u, N * mi .+ (1:p))
	else
		θ = view(x, N * ni + N * mi .+ (1:p))
	end

	for i = 1:N
		xi = view(x, (i - 1) * ni .+ (1:ni))
		ui = [u[(i - 1) * mi .+ (1:mi)]; θ]
		wi = view(w, (i - 1) * di .+ (1:di))
		push!(x⁺, fd(models.model, xi, ui, wi, h, t))
		push!(u_policy, policy(θ, xi, t, ni, mi))
	end

	return vcat(x⁺..., u_policy..., θ)
end

# Time
T = 11
h = 0.1
tf = h * (T - 1)
N = 5
models = multiple_model(model, T, N, p = p_policy)

x_ref = [[0.0; 0.0] for t = 1:T]
u_ref = [zeros(model.m) for t = 1:T-1]
_xT = [0.0; 0.0]
xT = [vcat([_xT for i = 1:N]..., zeros(t == 1 ? 0 : N * model.m), zeros(t == 1 ? 0 : models.p)) for t = 1:T]

# Initial conditions, controls, disturbances
x1 = zeros(models.n[1])
x1_add = [1.0, -1.0, 2.0, -2.0, 0.0]
for i = 1:N
	x1[(i - 1) * (model.n) + 1] = 0.0 + x1_add[i]
end


ū = [1.0 * randn(models.m[t]) for t = 1:T-1]
wi = [0.0, 0.0, 0.0, 0.0, 0.0]

@assert length(wi) == N
w = [vcat(wi...) for t = 1:T-1]

# Rollout
x̄ = rollout(models, x1, ū, w, h, T)

# Objective
_R = 1.0e-1 * ones(N * model.m)

Q = [(t < T ?
	 Diagonal(vcat([[1.0; 1.0] for i = 1:N]..., (t == 1 ? zeros(0) : _R)..., 1.0e-6 * ones(t == 1 ? 0 : models.p)))
	: Diagonal(vcat([[1.0; 1.0] for i = 1:N]..., (t == 1 ? zeros(0) : _R)..., 1.0e-6 * ones(t == 1 ? 0 : models.p)))) for t = 1:T]
q = [-2.0 * Q[t] * xT[t] for t = 1:T]

R = [Diagonal(t == 1 ? [_R; 1.0 * ones(models.p)] : _R) for t = 1:T-1]
r = [zeros(models.m[t]) for t = 1:T-1]

obj = StageCosts([QuadraticCost(Q[t], q[t],
	t < T ? R[t] : nothing, t < T ? r[t] : nothing) for t = 1:T], T)

function g(obj::StageCosts, x, u, t)
	T = obj.T
    if t < T
		Q = obj.cost[t].Q
		q = obj.cost[t].q
	    R = obj.cost[t].R
		r = obj.cost[t].r
        return (x' * Q * x + q' * x + u' * R * u + r' * u) ./ N
    elseif t == T
		Q = obj.cost[T].Q
		q = obj.cost[T].q
        return (x' * Q * x + q' * x) ./ N
    else
        return 0.0
    end
end

# Constraints
ns = models.N * models.model.n
ms = models.N * models.model.m
p_con = [t == T ? ns : ms for t = 1:T]
ul = [-Inf]
uu = [Inf]
info_t = Dict()
info_T = Dict(:xT => _xT)

con_set = [StageConstraint(p_con[t], t < T ? info_t : info_T) for t = 1:T]

function c!(c, cons::StageConstraints, x, u, t)
	T = cons.T
	N = models.N
	n = models.model.n
	m = models.model.m
	p = models.p
	np = n + p
	ns = N * n
	ms = N * m

	if t < T
		c[1:ms] = view(u, 1:ms)
	end

	if t == T
		for i = 1:N
			c[(i - 1) * n .+ (1:n)] .= view(x, (i - 1) * n .+ (1:n)) - cons.con[T].info[:xT]
		end
	end
end

prob = problem_data(models, obj, con_set, copy(x̄), copy(ū), w, h, T,
	n = models.n, m = models.m, d = models.d)

# Solve
@time constrained_ddp_solve!(prob,
    max_iter = 1000, max_al_iter = 10,
	ρ_init = 1.0, ρ_scale = 10.0,
	con_tol = 1.0e-3)

x, u = current_trajectory(prob)
x̄, ū = nominal_trajectory(prob)
x̄i = [x̄[t][1:model.n] for t = 1:T]

# individual trajectories
x_idx = [(i - 1) * model.n .+ (1:model.n) for i = 1:N]
u_idx = [N * model.n + (i - 1) * model.m .+ (1:model.m) for i = 1:N]

# Visualize
x_idxs = vcat(x_idx...)
u_idxs = vcat(u_idx...)

# state
plot(hcat([xT[t][x_idxs] for t = 1:T]...)',
    width = 2.0, color = :black, label = "")
plot!(hcat([x[t][x_idxs] for t = 1:T]...)', color = :magenta, label = "")

# control
plot(hcat([x[t][u_idxs] for t = 2:T]..., x[T][u_idxs], )',
    width = 2.0, color = :black, label = "", linetype = :steppost)

# verify solution
uθ = u[1][models.N * model.m .+ (1:models.p)]
xθ = [x[t][models.N * model.n + models.N * model.m .+ (1:models.p)] for t = 2:T]

policy_err = []
for t = 2:T
	push!(policy_err, norm(xθ[t-1] - uθ, Inf))
end
@show maximum(policy_err)

slack_err = []
for t = 1:T-1
	if t > 1
		push!(slack_err, norm(ū[t], Inf))
	else
		push!(slack_err, norm(ū[t][1:models.N * model.m], Inf))
	end
end
@show maximum(slack_err)

# Simulate policy
include(joinpath(@__DIR__, "simulate.jl"))

# Policy
θ = u[1][models.N * model.m .+ (1:models.p)]

# Model
model_sim = DoubleIntegratorContinuous{RK3, FixedTime}(model.n, model.m, model.d)
x1_sim = copy(x1[1:model.n])
T_sim = 10 * T

# Time
tf = h * (T - 1)
t = range(0, stop = tf, length = T)
t_sim = range(0, stop = tf, length = T_sim)
dt_sim = tf / (T_sim - 1)

# Simulate
N_sim = 10
x_sim = []
u_sim = []
J_sim = []
Random.seed!(1)
for k = 1:N_sim
	wi_sim = rand(range(-2.0, stop = 2.0, length = 1000))
	println("w: $wi_sim")
	# w_sim = [wi_sim for t = 1:T-1]
	w_sim = [wi_sim for t = 1:T-1]

	x_nn, u_nn, J_nn, Jx_nn, Ju_nn = simulate_policy(
		model_sim,
		θ,
		x_ref, u_ref,
		[_Q[1:model.n, 1:model.n] for _Q in Q], [_R[1:model.m, 1:model.m] for _R in R],
		T_sim, h,
		[wi_sim; 0.0],
		[zeros(model.d) for t = 1:T-1],
		ul = ul,
		uu = uu)

	push!(x_sim, x_nn)
	push!(u_sim, u_nn)
	push!(J_sim, J_nn)
end

# Visualize
idx = (1:2)
plt = plot(t, hcat(x_ref...)[idx, :]',
	width = 2.0, color = :black, label = "",
	xlabel = "time (s)", ylabel = "state",
	title = "double integrator (J_avg = $(round(mean(J_sim), digits = 3)), N_sim = $N_sim)")

for xs in x_sim
	plt = plot!(t_sim, hcat(xs...)[idx, :]',
	    width = 1.0, color = :magenta, label = "")
end
display(plt)

plt = plot(
	label = "",
	xlabel = "time (s)", ylabel = "control",
	title = "double integrator (J_avg = $(round(mean(J_sim), digits = 3)), N_sim = $N_sim)")
for us in u_sim
	plt = plot!(t_sim, hcat(us..., us[end])',
		width = 1.0, color = :magenta, label = "",
		linetype = :steppost)
end
display(plt)
