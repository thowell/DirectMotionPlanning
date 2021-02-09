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
d = model.d

struct MultipleModel{I, T} <: Model{I, T}
	n::Int
	m::Int
	d::Int

	model::Model

	N::Int
	p::Int
end

function multiple_model(model, N; p = 0)
	n = N * (model.n + p)
	m = N * model.m + p
	d = N * model.d
	MultipleModel{typeof(model).parameters...}(n, m, d, model, N, p)
end

N = 1
p_policy = model.n * model.n + model.n + model.m * model.n + model.m
models = multiple_model(model, N, p = p_policy)

function fd(models::MultipleModel, x, u, w, h, t)
	N = models.N
	p = models.p
	n = models.model.n
	n̄ = n + p
	m = models.model.m
	d = models.model.d

	x⁺ = []

	for i = 1:N
		xi = view(x, (i - 1) * n̄ .+ (1:n))
		ui = view(u, (i - 1) * m .+ (1:m))
		wi = view(w, (i - 1) * d .+ (1:d))

		if t == 1
			θ = view(u, N * m .+ (1:p))
		else
			θ = view(x, (i - 1) * n̄ + n .+ (1:p))
		end

		K1 = reshape(view(θ, 1:n * n), n, n)
		k1 = view(θ, n * n .+ (1:n))
		K2 = reshape(view(θ, n * n + n .+ (1:m * n)), m, n)
		k2 = view(θ, n * n + n + m * n .+ (1:m))

		z1 = tanh.(K1 * xi + k1)
		z2 = K2 * z1 + k2
		u_policy = ui + z2

		push!(x⁺, [fd(models.model, xi, u_policy, wi, h, t); θ])
	end

	return vcat(x⁺...)
end

# Time
T = 11
h = 0.1

z = range(0.0, stop = 3.0 * 2.0 * π, length = T)
p_ref = zeros(T)#1.0 * cos.(1.0 * z)
plot(z, p_ref)
xT = [vcat([[pr; 0.0; zeros(models.p)] for i = 1:N]...) for pr in p_ref]

# Initial conditions, controls, disturbances
# x1 = [p_ref[1]; 0.0; p_ref[1]; 0.0]
x1 = zeros(models.n)
for i = 1:N
	x1[(i - 1) * (model.n + models.p) + 1] = 1.0
end
x1

ū = [1.0e-3 * randn(models.m) for t = 1:T-1]
wi = [0.0]#, 0.5, 1.0]
@assert length(wi) == N
w = [vcat(wi...) for t = 1:T-1]

# Rollout
x̄ = rollout(models, x1, ū, w, h, T)

# Objective
Q = [(t < T ?
	h * Diagonal(vcat([[1.0; 1.0; 1.0e-5 * ones(models.p)] for i = 1:N]...))
	: Diagonal(vcat([[100.0; 100.0; 1.0e-5 * ones(models.p)] for i = 1:N]...))) for t = 1:T]
q = [-2.0 * Q[t] * xT[t] for t = 1:T]

R = [h * Diagonal(1.0e-1 * ones(models.m)) for t = 1:T-1]
r = [zeros(models.m) for t = 1:T-1]

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
1.0
# Constraints
p_con = [t == T ? 0 : (t > 1 ? models.m : (models.m - models.p)) for t = 1:T]
info_t = Dict()#:ul => [-5.0], :uu => [5.0], :inequality => (1:2 * m))
info_T = Dict()#:xT => xT)
con_set = [StageConstraint(p_con[t], t < T ? info_t : info_T) for t = 1:T]

function c!(c, cons::StageConstraints, x, u, t)
	T = cons.T

	if t == 1
		c .= view(u, 1:(models.m - models.p))
	elseif t < T
			c .= u
	else
		nothing
	end
end

prob = problem_data(models, obj, con_set, copy(x̄), copy(ū), w, h, T)

dynamics_derivatives!(prob.m_data)
prob.m_data.obj

prob.m_data.obj.costs.cost[1].Q
@time objective_derivatives!(prob.m_data.obj, prob.m_data)
prob.m_data.model.m
prob.m_data.model.n

# Solve
@time constrained_ddp_solve!(prob,
    max_iter = 1000, max_al_iter = 5,
	ρ_init = 1.0, ρ_scale = 10.0,
	con_tol = 1.0e-4)

@time ddp_solve!(prob,
    max_iter = 1000, verbose = true)

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

# plot(hcat(u..., u[end])[u_idxs, :]', linetype = :steppost)

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
