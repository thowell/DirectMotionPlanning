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

# Models
N = 2 * n + 1

# Time
T = 101
h = 0.1
t = range(0, stop = h * (T-1), length = T)

# Reference position trajectory
z = range(0.0, stop = 3.0 * 2.0 * π, length = T)
p_ref = 1.0 * cos.(1.0 * z)
# plot(z, p_ref)

# Initial conditions, controls, disturbances
x1 = [p_ref[1]; 0.0]
ū = [[0.01 * rand(model.m) for t = 1:T-1] for i = 1:N]

W = Distributions.MvNormal(zeros(model.d), Diagonal([0.0, 0.0, 0.0]))
w = [[rand(W, 1) for t = 1:T-1] for i = 1:N]

# Rollout
x̄ = [rollout(model, x1, ū[i], w[i], h, T) for i = 1:N]

p_idx = (1:1)
plt = plot()
for i = 1:N
	plt = plot!(t, hcat(x̄[i]...)[p_idx, :]',
		label = "")
end
display(plt)

# Objective
Q = Diagonal([100.0; 0.1])
R = Diagonal(0.01 * ones(model.m))

obj = StageCosts([QuadraticCost(Q, nothing,
	t < T ? R : nothing, nothing) for t = 1:T], T)

function g(obj::StageCosts, x, u, t)
	T = obj.T
    if t < T
		Q = obj.cost[t].Q
	    R = obj.cost[t].R
        return (x - [p_ref[t]; 0.0])' * Q * (x - [p_ref[t]; 0.0]) + u' * R * u
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
models_data = ModelData[]
for i = 1:N
	m_data = model_data(model, obj, w[i], h, T)
	m_data.x̄ .= x̄[i]
	m_data.ū .= ū[i]

	push!(models_data, m_data)
end
# objective(obj, x̄[4], ū[4])


# allocate policy data
p_data = policy_data(model, T)

# allocate solver data
s_data = solver_data(model, T)

# compute objective
function objective(data::ModelsData; mode = :nominal)
	N = length(data)
	J = 0.0

	for i = 1:N
		if mode == :nominal
			J += objective(data[i].obj, data[i].x̄, data[i].ū)
		elseif mode == :current
			J += objective(data[i].obj, data[i].x, data[i].u)
		else
			@error "objective - incorrect mode"
		end
	end

	return J
end

J = objective(models_data, mode = :nominal)

function derivatives!(data::ModelsData)
	N = length(data)

	for i = 1:N
		derivatives!(data[i])
	end
end

derivatives!(models_data)

function backward_pass!(p_data::PolicyData, models_data::ModelsData)
	N = length(models_data)
    T = models_data[1].T

    fx =  [m_data.dyn_deriv.fx for m_data in models_data]
    fu =  [m_data.dyn_deriv.fu for m_data in models_data]
    gx =  [m_data.obj_deriv.gx for m_data in models_data]
    gu =  [m_data.obj_deriv.gu for m_data in models_data]
    gxx = [m_data.obj_deriv.gxx for m_data in models_data]
    guu = [m_data.obj_deriv.guu for m_data in models_data]
    gux = [m_data.obj_deriv.gux for m_data in models_data]

    # policy
    K = p_data.K
    k = p_data.k

    # value function approximation
    P = p_data.P
    p = p_data.p

    # state-action value function approximation
    Qx = p_data.Qx
    Qu = p_data.Qu
    Qxx = p_data.Qxx
    Quu = p_data.Quu
    Qux = p_data.Qux

    # terminal value function
    P[T] = sum([gxx[i][T] for i = 1:N])
    p[T] = sum([gx[i][T] for i = 1:N])

    for t = T-1:-1:1
        Qx[t] =  sum([gx[i][t] + fx[i][t]' * p[t+1] for i = 1:N])
        Qu[t] =  sum([gu[i][t] + fu[i][t]' * p[t+1] for i = 1:N])
        Qxx[t] = sum([gxx[i][t] + fx[i][t]' * P[t+1] * fx[i][t] for i = 1:N])
        Quu[t] = sum([guu[i][t] + fu[i][t]' * P[t+1] * fu[i][t] for i = 1:N])
        Qux[t] = sum([gux[i][t] + fu[i][t]' * P[t+1] * fx[i][t] for i = 1:N])

        K[t] = -1.0 * Quu[t] \ Qux[t]
        k[t] = -1.0 * Quu[t] \ Qu[t]

        P[t] =  Qxx[t] + K[t]' * Quu[t] * K[t] + K[t]' * Qux[t] + Qux[t]' * K[t]
        p[t] =  Qx[t] + K[t]' * Quu[t] * k[t] + K[t]' * Qu[t] + Qux[t]' * k[t]
    end
end

backward_pass!(p_data, models_data)

function forward_pass!(p_data::PolicyData, models_data::ModelsData, s_data::SolverData;
    max_iter = 100)

	N = length(models_data)

    # reset solver status
    s_data.status = false

    # compute gradient of Lagrangian
    lagrangian_gradient!(s_data, p_data,
        models_data[1].model.n, models_data[1].model.m, models_data[1].T)

    # line search with rollout
    α = 1.0
    iter = 1
    while true
        iter > max_iter && (@error "forward pass failure", break)

        J = Inf
		i = 1

		while i <= N
	        try
	            rollout!(p_data, models_data[i], α = α)
	            # J = objective(m_data.obj, m_data.x, m_data.u)
	            Δz!(models_data[i])
				i += 1
	        catch
	            @warn "rollout failure (model $i)"
				fill!(models_data[i].z, 0.0)
				α *= 0.5
				iter += 1
				i = 1
	        end
		end
		println("$N rollouts successful: α = $α")
		J = objective(models_data, mode = :current)

        if J < s_data.obj + 0.001 * α * s_data.gradient' * sum([m.z for m in models_data])
            # update nominal
			for i = 1:N
	            models_data[i].x̄ .= deepcopy(models_data[i].x)
	            models_data[i].ū .= deepcopy(models_data[i].u)
			end
            s_data.obj = J
            s_data.status = true
            break
        else
            α *= 0.5
            iter += 1
        end
    end
end

s_data.obj = J

forward_pass!(p_data, models_data, s_data)
s_data.obj

function robust_solve(model::Model, obj::Objective, N, x̄, ū, w, h, T;
    max_iter = 10,
    grad_tol = 1.0e-5,
    verbose = true)

	println()
    (verbose && obj isa StageCosts) && println("Differential Dynamic Programming")

	# allocate model(s) data
	models_data = ModelData[]
	for i = 1:N
		m_data = model_data(model, obj, w[i], h, T)
		m_data.x̄ .= x̄[i]
		m_data.ū .= ū[i]

		push!(models_data, m_data)
	end

    # allocate policy data
    p_data = policy_data(model, T)

    # allocate solver data
    s_data = solver_data(model, T)

    # compute objective
    s_data.obj = objective(models_data, mode = :nominal)

    for i = 1:max_iter
        # derivatives
        derivatives!(models_data)

        # backward pass
        backward_pass!(p_data, m_data)

        # forward pass
        forward_pass!(p_data, m_data, s_data)

        # check convergence
        grad_norm = norm(s_data.gradient, Inf)
        verbose && println("     iter: $i
             cost: $(s_data.obj)
             grad norm: $(grad_norm)")
        (!s_data.status || grad_norm < grad_tol) && break
    end

    return p_data, m_data, s_data
end

# Solve
@time p_data, m_data, s_data = solve(model, obj, copy(x̄), copy(ū), w, h, T,
    max_iter = 100, verbose = true)


x = m_data.x
u = m_data.u

x̄ = m_data.x̄
ū = m_data.ū

# Visualize
using Plots
plot(hcat([[p_ref[t]; 0.0] for t = 1:T]...)',
    width = 2.0, color = :black, label = "")
plot!(hcat(x...)', color = :magenta, label = "")
# plot(hcat(u..., u[end])', linetype = :steppost)

# Simulate policy
using Random
Random.seed!(1)
include_dpo()
include(joinpath(pwd(), "examples/direct_policy_optimization/simulate.jl"))

# Model
model_sim = model
x1_sim = copy(x1)
T_sim = 10 * T

# Disturbance distributions
W = Distributions.MvNormal(zeros(model_sim.d),
	Diagonal([0.0, 0.0, 50.0]))
w = rand(W, T_sim)

W0 = Distributions.MvNormal(zeros(model_sim.d),
	Diagonal([0.0, 0.0, 0.0]))
w0 = rand(W0, 1)

# Initial state
z1_sim = vec(copy(x1_sim) + w0[1:2])

# Time
tf = h * (T - 1)
t = range(0, stop = tf, length = T)
t_sim = range(0, stop = tf, length = T_sim)
dt_sim = tf / (T_sim - 1)

# Policy
policy = linear_feedback(model.n, model.m)
# K, P = tvlqr(model, x̄, ū, h, [Q for t = 1:T], [R for t = 1:T-1])
K = [-K for K in p_data.K]

# Simulate
z_ddp, u_ddp, J_ddp, Jx_ddp, Ju_ddp = simulate(
	model_sim,
	linear_feedback(model.n, model.m), K,
    x̄, ū,
	[Q for t = 1:T], [R for t = 1:T-1],
	T_sim, h,
	z1_sim,
	w)

# Visualize
idx = (1:1)
plot(t, hcat(x̄...)[idx, :]',
    width = 2.0, color = :black, label = "")
plot!(t_sim, hcat(z_ddp...)[idx, :]',
    width = 1.0, color = :magenta, label = "")
