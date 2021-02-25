include(joinpath(pwd(), "examples/direct_policy_optimization/simulate.jl"))

using Random

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

tf_nom = sum([ū[t][end] for t = 1:T-1])
t_nom = range(0, stop = tf_nom, length = T)
t_sim_nom = range(0, stop = tf_nom, length = T_sim)

tf_dpo = sum([u[t][end] for t = 1:T-1])
t_nom_dpo = range(0, stop = tf_dpo, length = T)
t_sim_nom_dpo = range(0, stop = tf_dpo, length = T_sim)

dt_sim_nom = tf_nom / (T_sim - 1)
dt_sim_dpo = tf_dpo / (T_sim - 1)

# Simulate
N_sim = 100

Random.seed!(1)
W0 = Distributions.MvNormal(zeros(model_sim.n),
	Diagonal(1.0e-3 * ones(model_sim.n)))
w0 = rand(W0, N_sim)

z_lqr1 = []
u_lqr1 = []
J_lqr1 = []
Jx_lqr1 = []
Ju_lqr1 = []

for i = 1:N_sim
	z0_sim = vec(copy(x1_sim) + w0[:, i])
	println("sim: $i")
	@show z0_sim[1:3]
	_z_lqr1, _u_lqr1, _J_lqr1, _Jx_lqr1, _Ju_lqr1 = simulate(
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

	push!(z_lqr1, _z_lqr1)
	push!(u_lqr1, _u_lqr1)
	push!(J_lqr1, _J_lqr1)
	push!(Jx_lqr1, _Jx_lqr1)
	push!(Ju_lqr1, _Ju_lqr1)
end

@show mean(J_lqr1)
@show std(J_lqr1)
@show mean(Jx_lqr1)
@show std(Jx_lqr1)
@show mean(Ju_lqr1)
@show std(Ju_lqr1)

z_lqr2 = []
u_lqr2 = []
J_lqr2 = []
Jx_lqr2 = []
Ju_lqr2 = []

for i = 1:N_sim
	z0_sim = vec(copy(x1_sim) + w0[:, i])
	println("sim: $i")
	@show z0_sim[1:3]

	_z_lqr2, _u_lqr2, _J_lqr2, _Jx_lqr2, _Ju_lqr2 = simulate(
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

	push!(z_lqr2, _z_lqr2)
	push!(u_lqr2, _u_lqr2)
	push!(J_lqr2, _J_lqr2)
	push!(Jx_lqr2, _Jx_lqr2)
	push!(Ju_lqr2, _Ju_lqr2)
end

@show mean(J_lqr2)
@show std(J_lqr2)
@show mean(Jx_lqr2)
@show std(Jx_lqr2)
@show mean(Ju_lqr2)
@show std(Ju_lqr2)

z_lqr3 = []
u_lqr3 = []
J_lqr3 = []
Jx_lqr3 = []
Ju_lqr3 = []

for i = 1:N_sim
	z0_sim = vec(copy(x1_sim) + w0[:, i])
	println("sim: $i")
	@show z0_sim[1:3]

	_z_lqr3, _u_lqr3, _J_lqr3, _Jx_lqr3, _Ju_lqr3 = simulate(
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

	push!(z_lqr3, _z_lqr3)
	push!(u_lqr3, _u_lqr3)
	push!(J_lqr3, _J_lqr3)
	push!(Jx_lqr3, _Jx_lqr3)
	push!(Ju_lqr3, _Ju_lqr3)
end
idx_finite = isfinite.(J_lqr3)
@show mean(J_lqr3[idx_finite])
@show std(J_lqr3[idx_finite])
@show mean(Jx_lqr3[idx_finite])
@show std(Jx_lqr3[idx_finite])
@show mean(Ju_lqr3[idx_finite])
@show std(Ju_lqr3[idx_finite])

z_lqr4 = []
u_lqr4 = []
J_lqr4 = []
Jx_lqr4 = []
Ju_lqr4 = []

for i = 1:N_sim
	z0_sim = vec(copy(x1_sim) + w0[:, i])
	println("sim: $i")
	@show z0_sim[1:3]

	_z_lqr4, _u_lqr4, _J_lqr4, _Jx_lqr4, _Ju_lqr4 = simulate(
		model_sim,
		policy, K,
	    x̄, ū,
		Q, R,
		T_sim, ū[1][end],
		z0_sim, w,
		_norm = 2,
		ul = ul_nom, uu = uu4,
		u_idx = (1:model.m - 1))

	push!(z_lqr4, _z_lqr4)
	push!(u_lqr4, _u_lqr4)
	push!(J_lqr4, _J_lqr4)
	push!(Jx_lqr4, _Jx_lqr4)
	push!(Ju_lqr4, _Ju_lqr4)
end
idx_finite = isfinite.(J_lqr4)
convert.(Int, range(1, stop = 100, length = 100)[.!isfinite.(J_lqr4)])
@show mean(J_lqr4[idx_finite])
@show std(J_lqr4[idx_finite])
@show mean(Jx_lqr4[idx_finite])
@show std(Jx_lqr4[idx_finite])
@show mean(Ju_lqr4[idx_finite])
@show std(Ju_lqr4[idx_finite])

z_dpo1 = []
u_dpo1 = []
J_dpo1 = []
Jx_dpo1 = []
Ju_dpo1 = []

for i = 1:N_sim
	z0_sim = vec(copy(x1_sim) + w0[:, i])
	println("sim: $i")
	@show z0_sim[1:3]

	_z_dpo1, _u_dpo1, _J_dpo1, _Jx_dpo1, _Ju_dpo1 = simulate(
		model_sim,
		policy, θ,
	    x, u,
		Q, R,
		T_sim, u[1][end],
		z0_sim, w,
		_norm = 2,
		ul = ul_nom, uu = uu1,
		u_idx = (1:model.m - 1))

	push!(z_dpo1, _z_dpo1)
	push!(u_dpo1, _u_dpo1)
	push!(J_dpo1, _J_dpo1)
	push!(Jx_dpo1, _Jx_dpo1)
	push!(Ju_dpo1, _Ju_dpo1)
end

@show mean(J_dpo1)
@show std(J_dpo1)
@show mean(Jx_dpo1)
@show std(Jx_dpo1)
@show mean(Ju_dpo1)
@show std(Ju_dpo1)

z_dpo2 = []
u_dpo2 = []
J_dpo2 = []
Jx_dpo2 = []
Ju_dpo2 = []

for i = 1:N_sim
	z0_sim = vec(copy(x1_sim) + w0[:, i])
	println("sim: $i")
	@show z0_sim[1:3]
	_z_dpo2, _u_dpo2, _J_dpo2, _Jx_dpo2, _Ju_dpo2 = simulate(
		model_sim,
		policy, θ,
	    x, u,
		Q, R,
		T_sim, u[1][end],
		z0_sim, w,
		_norm = 2,
		ul = ul_nom, uu = uu2,
		u_idx = (1:model.m - 1))

	push!(z_dpo2, _z_dpo2)
	push!(u_dpo2, _u_dpo2)
	push!(J_dpo2, _J_dpo2)
	push!(Jx_dpo2, _Jx_dpo2)
	push!(Ju_dpo2, _Ju_dpo2)
end

@show mean(J_dpo2)
@show std(J_dpo2)
@show mean(Jx_dpo2)
@show std(Jx_dpo2)
@show mean(Ju_dpo2)
@show std(Ju_dpo2)

z_dpo3 = []
u_dpo3 = []
J_dpo3 = []
Jx_dpo3 = []
Ju_dpo3 = []

for i = 1:N_sim
	z0_sim = vec(copy(x1_sim) + w0[:, i])
	println("sim: $i")
	@show z0_sim[1:3]

	_z_dpo3, _u_dpo3, _J_dpo3, _Jx_dpo3, _Ju_dpo3 = simulate(
		model_sim,
		policy, θ,
	    x, u,
		Q, R,
		T_sim, u[1][end],
		z0_sim, w,
		_norm = 2,
		ul = ul_nom, uu = uu3,
		u_idx = (1:model.m - 1))
	push!(z_dpo3, _z_dpo3)
	push!(u_dpo3, _u_dpo3)
	push!(J_dpo3, _J_dpo3)
	push!(Jx_dpo3, _Jx_dpo3)
	push!(Ju_dpo3, _Ju_dpo3)
end

@show mean(J_dpo3)
@show std(J_dpo3)
@show mean(Jx_dpo3)
@show std(Jx_dpo3)
@show mean(Ju_dpo3)
@show std(Ju_dpo3)

z_dpo4 = []
u_dpo4 = []
J_dpo4 = []
Jx_dpo4 = []
Ju_dpo4 = []

for i = 1:N_sim
	z0_sim = vec(copy(x1_sim) + w0[:, i])
	println("sim: $i")
	@show z0_sim[1:3]

	_z_dpo4, _u_dpo4, _J_dpo4, _Jx_dpo4, _Ju_dpo4 = simulate(
		model_sim,
		policy, θ,
	    x, u,
		Q, R,
		T_sim, u[1][end],
		z0_sim, w,
		_norm = 2,
		ul = ul_nom, uu = uu4,
		u_idx = (1:model.m - 1))
	push!(z_dpo4, _z_dpo4)
	push!(u_dpo4, _u_dpo4)
	push!(J_dpo4, _J_dpo4)
	push!(Jx_dpo4, _Jx_dpo4)
	push!(Ju_dpo4, _Ju_dpo4)
end

@show mean(J_dpo4)
@show std(J_dpo4)
@show mean(Jx_dpo4)
@show std(Jx_dpo4)
@show mean(Ju_dpo4)
@show std(Ju_dpo4)
