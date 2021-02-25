include(joinpath(pwd(), "examples/direct_policy_optimization/simulate.jl"))

# Unpack trajectories
x̄, ū = unpack(z̄, prob)
x, u = unpack(z, prob)
θ = get_policy(z, prob_dpo)

# Simulation setup
model_sim = Quadrotor{RK3, FixedTime}(n, m, d,
                  mass,
                  J,
                  Jinv,
                  g,
                  L,
                  kf,
                  km)

x1_sim = copy(x1)
T_sim = 10 * T

using Random
Random.seed!(1)

W = Distributions.MvNormal(zeros(model_sim.n),
	Diagonal(1.0e-5 * ones(model_sim.n)))
w = rand(W, T_sim)

W0 = Distributions.MvNormal(zeros(model_sim.n),
	Diagonal(1.0e-2 * ones(model_sim.n)))
w0 = rand(W0, 1)

z0_sim = vec(copy(x1_sim) + w0)


tf_nom = sum([ū[t][end] for t = 1:T-1])
t_nom = range(0, stop = tf_nom, length = T)
t_sim_nom = range(0, stop = tf_nom, length = T_sim)

tf_dpo = sum([u[t][end] for t = 1:T-1])
t_nom_dpo = range(0, stop = tf_dpo, length = T)
t_sim_nom_dpo = range(0, stop = tf_dpo, length = T_sim)

dt_sim_nom = tf_nom / (T_sim - 1)
dt_sim_dpo = tf_dpo / (T_sim - 1)

# Simulate
z_lqr1, u_lqr1, J_lqr1, Jx_lqr1, Ju_lqr1 = simulate(
	model_sim,
	policy, K,
    x̄, ū,
	Q, R,
	T_sim, ū[1][end],
	z0_sim,
	w,
	_norm = 2,
	ul = ul_nom, uu = uu1,
	u_idx = (1:model.m - 1))

z_lqr2, u_lqr2, J_lqr2, Jx_lqr2, Ju_lqr2 = simulate(
	model_sim,
	policy,
	K,
    x̄, ū,
	Q, R,
	T_sim, ū[1][end],
	z0_sim,
	w,
	_norm = 2,
	ul = ul_nom,
	uu = uu2,
	u_idx = (1:model.m - 1))

z_lqr3, u_lqr3, J_lqr3, Jx_lqr3, Ju_lqr3 = simulate(
	model_sim,
	policy, K,
    x̄, ū,
	Q, R,
	T_sim, ū[1][end],
	z0_sim,
	w,
	_norm = 2,
	ul = ul_nom, uu = uu3,
	u_idx = (1:model.m - 1))

z_lqr4, u_lqr4, J_lqr4, Jx_lqr4, Ju_lqr4 = simulate(
	model_sim,
	policy, K,
    x̄, ū,
	Q, R,
	T_sim, ū[1][end],
	z0_sim, w,
	_norm = 2,
	ul = ul_nom, uu = uu4,
	u_idx = (1:model.m - 1))

z_dpo1, u_dpo1, J_dpo1, Jx_dpo1, Ju_dpo1 = simulate(
	model_sim,
	policy, θ,
    x, u,
	Q, R,
	T_sim, u[1][end],
	z0_sim, w,
	_norm = 2,
	ul = ul_nom, uu = uu1,
	u_idx = (1:model.m - 1))

z_dpo2, u_dpo2, J_dpo2, Jx_dpo2, Ju_dpo2 = simulate(
	model_sim,
	policy, θ,
    x, u,
	Q, R,
	T_sim, u[1][end],
	z0_sim, w,
	_norm = 2,
	ul = ul_nom, uu = uu2,
	u_idx = (1:model.m - 1))

z_dpo3, u_dpo3, J_dpo3, Jx_dpo3, Ju_dpo3 = simulate(
	model_sim,
	policy, θ,
    x, u,
	Q, R,
	T_sim, u[1][end],
	z0_sim, w,
	_norm = 2,
	ul = ul_nom, uu = uu3,
	u_idx = (1:model.m - 1))

z_dpo4, u_dpo4, J_dpo4, Jx_dpo4, Ju_dpo4 = simulate(
	model_sim,
	policy, θ,
    x, u,
	Q, R,
	T_sim, u[1][end],
	z0_sim, w,
	_norm = 2,
	ul = ul_nom, uu = uu4,
	u_idx = (1:model.m - 1))

# state tracking
(Jx_lqr1 + Jx_lqr2 + Jx_lqr3 + Jx_lqr4) / 4.0
(Jx_dpo1 + Jx_dpo2 + Jx_dpo3 + Jx_dpo4) / 4.0

# control tracking
(Ju_lqr1 + Ju_lqr2 + Ju_lqr3 + Ju_lqr4) / 4.0
(Ju_dpo1 + Ju_dpo2 + Ju_dpo3 + Ju_dpo4) / 4.0

# objective value
(J_lqr1 + J_lqr2 + J_lqr3 + J_lqr4) / 4.0
(J_dpo1 + J_dpo2 + J_dpo3 + J_dpo4) / 4.0
