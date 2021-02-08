using Plots
using Random
Random.seed!(1)

include_ddp()

# Model
include_model("double_integrator")

function f(model::DoubleIntegratorContinuous, x, u, w)
    [x[2] + w[1]; (1.0 + w[3]) * u[1] + w[2]]
end

model = DoubleIntegratorContinuous{Midpoint, FixedTime}(2, 1, 3)
n = model.n
m = model.m

struct MultipleModel{I, T} <: Model{I, T}
	n::Int
	m::Int
	d::Int

	N::Int
	model::Model
	p::Int
end

function multiple_model(model, N; p = 0)
	n = N * (model.n + p)
	m = N * model.m + p
	d = N * model.d
	MultipleModel{typeof(model).parameters...}(n, m, d, N, model, p)
end

N = 2
p_policy = model.n * model.n + model.n + model.m * model.n + model.m
models = multiple_model(model, N, p = p_policy)

function fd(model::DoubleIntegratorContinuous{Midpoint, FixedTime}, x, u, w, h, t)
	# if t == 1
	# 	x1 = u[2:3]
    # 	return x1 + h * f(model, x1 + 0.5 * h * f(model, x1, u, w), u, w)
	# else
	return x + h * f(model, x + 0.5 * h * f(model, x, u, w), u, w)
	# end
end

function fd(model::MultipleModel, x, u, w, h, t)
	N = model.N
	p = model.p
	n = model.model.n
	n̄ = n + p
	m = model.model.m
	d = model.model.d

	x⁺ = []

	for i = 1:N
		xi = view(x, (i - 1) * n̄ .+ (1:n))
		ui = view(u, (i - 1) * m .+ (1:m))
		wi = view(w, (i - 1) * d .+ (1:d))

		if t == 1
			θ = u[N * m .+ (1:p)]
		else
			θ = view(x, (i - 1) * n̄ + n .+ (1:p))
		end

		K1 = reshape(θ[1:n * n], n, n)
		k1 = θ[n * n .+ (1:n)]
		K2 = reshape(θ[n * n + n .+ (1:m * n)], m, n)
		k2 = θ[n * n + n + m * n .+ (1:m)]

		z1 = tanh.(K1 * xi + k1)
		z2 = K2 * z1 + k2
		u_policy = ui + z2

		push!(x⁺, [fd(model.model, xi, u_policy, wi, h, t); θ])
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
# w = [[0.0; 0.0; 0.0; 0.0; 0.0; 1.0] for t = 1:T-1]
w = [[0.0; 0.0; 0.0; 0.0; 0.0; 1.0] for t = 1:T-1]

# Rollout
x̄ = rollout(models, x1, ū, w, h, T)

# Objective
Q = [t < T ? h * Diagonal(vcat([[1.0; 1.0; 1.0e-5 * ones(models.p)] for i = 1:N]...)) : Diagonal(vcat([[100.0; 100.0; 1.0e-5 * ones(models.p)] for i = 1:N]...)) for t = 1:T]
R = [h * Diagonal(1.0e-1 * ones(models.m)) for t = 1:T-1]

obj = StageCosts([QuadraticCost(Q[t], nothing,
	t < T ? R[t] : nothing, nothing) for t = 1:T], T)

function g(obj::StageCosts, x, u, t)
	T = obj.T
    if t < T
		Q = obj.cost[t].Q
	    R = obj.cost[t].R
        return h * (x - xT[t])' * Q * (x - xT[t]) + u' * R * u
    elseif t == T
		Q = obj.cost[T].Q
        return (x - xT[t])' * Q * (x - xT[t])
    else
        return 0.0
    end
end

g(obj, x̄[T-1], ū[T-1], T-1)
g(obj, x̄[T], nothing, T)
objective(obj, x̄, ū)

# Constraints
p_con = [t == T ? 0 : (t > 1 ? models.m : (models.m - models.p)) for t = 1:T]
info_t = Dict()#:ul => [-5.0], :uu => [5.0], :inequality => (1:2 * m))
info_T = Dict()#:xT => xT)
con_set = [StageConstraint(p_con[t], t < T ? info_t : info_T) for t = 1:T]

function c!(c, cons::StageConstraints, x, u, t)
	T = cons.T

	if t == 1
		c .= u[1:(models.m - models.p)]
	elseif t < T
			c .= u
	else
		nothing
	end
end

prob = problem_data(models, obj, con_set, copy(x̄), copy(ū), w, h, T)

# Solve
@time constrained_ddp_solve!(prob,
    max_iter = 1000, max_al_iter = 5,
	ρ_init = 1.0, ρ_scale = 10.0,
	con_tol = 1.0e-3)

# prob = problem_data(models, obj, x̄, ū, w, h, T)
#
# # Solve
# @time ddp_solve!(prob,
#     max_iter = 100, verbose = true)

x, u = current_trajectory(prob)
x̄, ū = nominal_trajectory(prob)
prob.m_data
u[1][1:2]
# Visualize
models.p
idx = collect([(1:2)..., (model.n + models.p .+ (1:2))...])
using Plots
plot(hcat([xT[t] for t = 1:T]...)[idx, :]',
    width = 2.0, color = :black, label = "")
plot!(hcat(x...)[idx, :]', color = :magenta, label = "")
plot(hcat(u..., u[end])', linetype = :steppost)
u[1][3:5]
u[1]
u[2]

u[1][1:2]
u[1][3:5]
x[2][2 .+ (1:3)]
