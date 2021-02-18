using Plots
using Random
Random.seed!(1)

# ddp
include_ddp()

# Model
include_model("pendulum")

model = Pendulum{RK3, FixedTime}(2, 1, 1, 1.0, 0.1, 0.5, 9.81)
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

	n = [N * n + (t != 1 ? N * m : 0) for t = 1:T]
	m = [N * m + (t == 1 ? p : p) for t = 1:T-1]
	d = [N * d for t = 1:T-1]

	MultipleModel{typeof(model).parameters...}(n, m, d, model, N, p)
end

# Policy

p_policy = model.m * model.n + model.m
# p_policy = dp * model.n * model.n + dp * model.n + model.m * dp * model.n + model.m
# p_policy = model.m * model.n + model.m

function policy(θ, x, t, n, m)

	K = reshape(view(θ, 1:(m * n)), m, n)
	k = view(θ, m * n .+ (1:m))

	u = K * x + k

	return u
end

function f(model::Pendulum, x, u, w)
	mass = model.mass + w[1]
    @SVector [x[2],
              ((u[1] + policy(view(u, 1 .+ (1:p_policy)), x, nothing, model.n, model.m)[1]) / ((mass * model.lc * model.lc))
                - model.g * sin(x[1]) / model.lc
                - model.b * x[2] / (mass * model.lc * model.lc))]
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

	θ = view(u, N * mi .+ (1:p))

	for i = 1:N
		xi = view(x, (i - 1) * ni .+ (1:ni))
		ui = [u[(i - 1) * mi .+ (1:mi)]; θ]
		wi = view(w, (i - 1) * di .+ (1:di))
		push!(x⁺, fd(models.model, xi, ui, wi, h, t))
		push!(u_policy, policy(θ, xi, t, ni, mi))
	end

	return vcat(x⁺..., u_policy...)
end

# Time
T = 51
h = 0.1
tf = h * (T - 1)
N = 5
models = multiple_model(model, T, N, p = p_policy)

x_ref = [[π; 0.0] for t = 1:T]
u_ref = [zeros(model.m) for t = 1:T-1]
_xT = [π; 0.0]
xT = [vcat([_xT for i = 1:N]..., zeros(t == 1 ? 0 : N * model.m)) for t = 1:T]

# Initial conditions, controls, disturbances
x1 = zeros(models.n[1])
# x1_add = [1.0, -1.0, 2.0, -2.0, 0.0]
# for i = 1:N
# 	x1[(i - 1) * (model.n) + 1] = 0.0 + x1_add[i]
# end

ū = [1.0 * randn(models.m[t]) for t = 1:T-1]
wi = [0.0, 0.1, -0.1, 0.2, -0.2]#, 0.0, 0.0, 0.0, 0.0]

@assert length(wi) == N
w = [vcat(wi...) for t = 1:T-1]

# Rollout
x̄ = rollout(models, x1, ū, w, h, T)

# Objective
_R = 1.0e-1 * ones(N * model.m)

Q = [(t < T ?
	 Diagonal(vcat([[1.0; 1.0] for i = 1:N]..., (t == 1 ? zeros(0) : _R)...))
	: Diagonal(vcat([[1.0; 1.0] for i = 1:N]..., (t == 1 ? zeros(0) : _R)...))) for t = 1:T]
q = [-2.0 * Q[t] * xT[t] for t = 1:T]

R = [Diagonal([_R; 1.0 * ones(models.p)]) for t = 1:T-1]
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
		# c[1:ms] = view(u, 1:ms)
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
    max_iter = 1000, max_al_iter = 20,
	ρ_init = 1.0, ρ_scale = 10.0, ρ_max = Inf,
	con_tol = 1.0e-5)

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
θ = [ū[t][models.N * model.m .+ (1:models.p)] for t = 1:T-1]

# x_sim = [x1]
# u_sim = []
# for t = 1:T-1
# 	push!(u_sim, policy(θ[t], x_sim[end], nothing, model.n, model.m))
# 	push!(x_sim, fd(model, x_sim[end], [zeros(model.m); θ[t]], zeros(model.d), h, t))
# end

slack_err = []
for t = 1:T-1
	if t > 1
		push!(slack_err, norm(ū[t][1:models.N * model.m], Inf))
	else
		push!(slack_err, norm(ū[t][1:models.N * model.m], Inf))
	end
end
@show maximum(slack_err)

# Simulate policy
include(joinpath(@__DIR__, "simulate.jl"))

# Model
model_sim = Pendulum{RK3, FixedTime}(2, 1, 1, 1.0, 0.1, 0.5, 9.81)
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
	wi_sim = 0.0 * rand(range(-2.0, stop = 2.0, length = 1000))
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
		[zeros(model.d) for t = 1:T-1])#,
		# ul = ul,
		# uu = uu)

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
