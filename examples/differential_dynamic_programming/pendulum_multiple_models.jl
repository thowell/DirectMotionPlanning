using Plots
using Random
Random.seed!(1)

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

	# n = [N * (n + (t == 1 ? 0 : p)) for t = 1:T]
	n = [N * (n + p) for t = 1:T]
	m = [N * m + (t == 1 ? p : 0) for t = 1:T-1]
	d = [N * d for t = 1:T-1]

	MultipleModel{typeof(model).parameters...}(n, m, d, model, N, p)
end

# Policy
p_policy = model.n * model.n + model.n + model.m * model.n + model.m

function policy(θ, x, n, m)
	K1 = reshape(view(θ, 1:n * n), n, n)
	k1 = view(θ, n * n .+ (1:n))
	K2 = reshape(view(θ, n * n + n .+ (1:m * n)), m, n)
	k2 = view(θ, n * n + n + m * n .+ (1:m))

	z1 = tanh.(K1 * x + k1)
	# z2 = tanh.(K1 * z1 + k1)
	# z3 = tanh.(K1 * z2 + k1)
	zo = K2 * z1 + k2

	return zo
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

	nip = ni + p

	x⁺ = []

	for i = 1:N
		xi = view(x, (i - 1) * nip .+ (1:ni))
		ui = view(u, (i - 1) * mi .+ (1:mi))
		wi = view(w, (i - 1) * di .+ (1:di))

		if t == 1
			θ = view(u, N * mi .+ (1:p))
		else
			θ = view(x, (i - 1) * nip + ni .+ (1:p))
		end

		u_ctrl = ui + policy(θ, xi, ni, mi)

		push!(x⁺, [fd(models.model, xi, u_ctrl, wi, h, t); θ])
	end

	return vcat(x⁺...)
end

# Time
T = 101
h = 0.025

N = 2 * model.n + 1
models = multiple_model(model, T, N, p = p_policy)

x1 = zeros(models.n[1])
_xT = [π, 0.0] # goal state
xT = [vcat([[_xT; zeros(models.p)] for i = 1:N]...) for t = 1:T]

ū = [1.0e-1 * randn(models.m[t]) for t = 1:T-1]
wi = [0.15, -0.15, 0.3, -0.3, 0.0]#, 0.025, 0.05, 0.075, 0.1]#, 0.5, 1.0]
@assert length(wi) == N
w = [vcat(wi...) for t = 1:T-1]

# Rollout
x̄ = rollout(models, x1, ū, w, h, T)

# Objective
Q = [(t < T ?
	 Diagonal(vcat([[1.0; 1.0; 1.0e-5 * ones(models.p)] for i = 1:N]...))
	: Diagonal(vcat([[1000.0; 1000.0; 1.0e-5 * ones(models.p)] for i = 1:N]...))) for t = 1:T]
q = [-2.0 * Q[t] * xT[t] for t = 1:T]

R = [Diagonal(1.0e-1 * ones(models.m[t])) for t = 1:T-1]
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
        return x' * Q * x + q' * x + u' * R * u + r' * u
    elseif t == T
		Q = obj.cost[T].Q
		q = obj.cost[T].q
        return x' * Q * x + q' * x
    else
        return 0.0
    end
end

g(obj, x̄[T-1], ū[T-1], T-1)
g(obj, x̄[T], nothing, T)
objective(obj, x̄, ū)

# Constraints
p_con = [t == T ? 0 : (t > 1 ? models.m[t] : (models.m[t] - models.p)) for t = 1:T]
info_t = Dict()#:ul => [-5.0], :uu => [5.0], :inequality => (1:2 * m))
info_T = Dict()#:xT => xT)
con_set = [StageConstraint(p_con[t], t < T ? info_t : info_T) for t = 1:T]

function c!(c, cons::StageConstraints, x, u, t)
	T = cons.T

	if t == 1
		c .= view(u, 1:(models.m[t] - models.p))
	elseif t < T
		c .= u
	else
		nothing
	end
end

prob = problem_data(models, obj, con_set, copy(x̄), copy(ū), w, h, T,
	n = models.n, m = models.m, d = models.d)

# Solve
@time constrained_ddp_solve!(prob,
    max_iter = 1000, max_al_iter = 8,
	ρ_init = 1.0, ρ_scale = 10.0,
	con_tol = 1.0e-5)

x, u = current_trajectory(prob)
x̄, ū = nominal_trajectory(prob)

# individual trajectories
x_idx = [(i - 1) * (model.n + models.p) .+ (1:model.n) for i = 1:N]
u_idx = [(i - 1) * model.m .+ (1:model.m) for i = 1:N]

# Visualize

x_idxs = vcat(x_idx...)
u_idxs = vcat(u_idx...)

# state
plot(hcat([xT[t] for t = 1:T]...)[x_idxs, :]',
    width = 2.0, color = :black, label = "")
plot!(hcat(x...)[x_idxs, :]', color = :magenta, label = "")

# verify solution
uθ = u[1][models.N * model.m .+ (1:models.p)]
xθ_idx = [(i - 1) * (model.n + models.p) + model.n .+ (1:models.p) for i = 1:N]

policy_err = []
for i = 1:N
	for t = 2:T
		push!(policy_err, norm(x̄[t][xθ_idx[i]] - uθ, Inf))
	end
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

# Model
model_sim = model
x1_sim = copy(x1)
T_sim = 10 * T
w_sim = [[0.0] for t = 1:T-1]

# Time
tf = h * (T - 1)
t = range(0, stop = tf, length = T)
t_sim = range(0, stop = tf, length = T_sim)
dt_sim = tf / (T_sim - 1)

# Simulate
x_ddp, u_ddp, J_ddp, Jx_ddp, Ju_ddp = simulate_policy(
	model_sim,
	uθ,
    [x[1:model.n] for x in x̄], [u[1:model.m] for u in ū],
	[_Q[1:model.n, 1:model.n] for _Q in Q], [_R[1:model.m, 1:model.m] for _R in R],
	T_sim, h,
	x1_sim,
	w_sim)

# Visualize
idx = (1:2)
plot(t, hcat(x̄...)[idx, :]',
    width = 2.0, color = :black, label = "")
plot!(t_sim, hcat(x_ddp...)[idx, :]',
    width = 1.0, color = :magenta, label = "")

plot(t, hcat(u..., u[end])',
	width = 2.0, color = :black, label = "",
	linetype = :steppost)
plot!(t_sim, hcat(u_ddp..., u_ddp[end])',
	width = 1.0, color = :magenta, label = "",
	linetype = :steppost)
