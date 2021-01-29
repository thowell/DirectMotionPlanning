include_ddp()

# Model
include_model("double_integrator")

function f(model::DoubleIntegratorContinuous, x, u, w)
    [x[2] + w[1]; (1.0 + w[3]) * u[1] + w[2]]
end

model = DoubleIntegratorContinuous{Midpoint, FixedTime}(2, 1, 3)
n = model.n
m = model.m

# Time
T = 101
h = 0.1

z = range(0.0, stop = 3.0 * 2.0 * π, length = T)
p_ref = 1.0 * cos.(1.0 * z)
plot(z, p_ref)

# Initial conditions, controls, disturbances
x1 = [p_ref[1]; 0.0]
ū = [rand(model.m) for t = 1:T-1]
w = [zeros(model.d) for t = 1:T-1]

# Rollout
x̄ = rollout(model, x1, ū, w, h, T)

# Objective
Q = Diagonal([10.0; 0.1])
R = Diagonal(0.1 * ones(model.m))

obj = StageQuadratic(Q, nothing, R, nothing, T)

function g(obj::StageQuadratic, x, u, t)
    Q = obj.Q
    R = obj.R
    T = obj.T

    if t < T
        return (x - [p_ref[t]; 0.0])' * Q * (x - [p_ref[t]; 0.0]) + u' * R * u
    elseif t == T
        return (x - [p_ref[T]; 0.0])' * Q * (x - [p_ref[T]; 0.0])
    else
        return 0.0
    end
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
plot(hcat([[x_ref[t]; 0.0] for t = 1:T]...)',
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
	Diagonal([5.0, 5.0, 15.0]))
w = rand(W, T_sim)

W0 = Distributions.MvNormal(zeros(model_sim.d),
	Diagonal([0.1, 0.1, 0.0]))
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
plot(t, hcat(x̄...)',
    width = 2.0, color = :black, label = "")
plot!(t_sim, hcat(z_ddp...)',
    width = 1.0, color = :magenta, label = "")
