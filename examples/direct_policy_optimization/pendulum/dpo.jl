include_dpo()
include(joinpath(@__DIR__, "pendulum_minimum_time.jl"))

function fd(model::Pendulum{Midpoint, FreeTime}, x⁺, x, u, w, h, t)
	h = u[end]
    x⁺ - (x + h * f(model, 0.5 * (x + x⁺), u, w) + w)
end

# Nominal solution
x̄, ū = unpack(z̄, prob)
prob_nom = prob.prob

# DPO
β = 1.0
δ = 5.0e-3

# Initial samples
x1_sample = resample(x1, Diagonal([1.0; 1.0]), 1.0e-3)

# Mean problem
prob_mean = trajectory_optimization(
				model,
				EmptyObjective(),
				T,
				ul = control_bounds(model, T, [Inf; 0.0], [Inf; 0.0])[1],
				uu = control_bounds(model, T, [Inf; 0.0], [Inf; 0.0])[2],
				dynamics = false)

# Sample problems
prob_sample = [trajectory_optimization(
				model,
				EmptyObjective(),
				T,
				xl = state_bounds(model, T, x1 = x1_sample[i])[1],
				xu = state_bounds(model, T, x1 = x1_sample[i])[2],
				ul = ul,
				uu = uu,
				dynamics = false,
				con = con_free_time) for i = 1:2 * model.n]

# Sample objective
Q = [(t < T ? Diagonal(10.0 * ones(model.n))
	: Diagonal(100.0 * ones(model.n))) for t = 1:T]
R = [Diagonal(1.0e-1 * [1.0; 10.0]) for t = 1:T-1]

obj_sample = sample_objective(Q, R)
policy = linear_feedback(model.n, model.m - 1)
dist = disturbances([Diagonal(δ * ones(model.d)) for t = 1:T-1])
sample = sample_params(β, T)

prob_dpo = dpo_problem(
	prob_nom, prob_mean, prob_sample,
	obj_sample,
	policy,
	dist,
	sample)

# TVLQR policy
K, P = tvlqr(model, x̄, ū, 0.0, Q, R)

z0 = pack(z̄, K, prob_dpo)

# Solve
if false
	include_snopt() # set path
	z, info = solve(prob_dpo, copy(z0),
		nlp = :SNOPT7,
		tol = 1.0e-3, c_tol = 1.0e-3,
		time_limit = 60 * 2)
	@save joinpath(@__DIR__, "sol_dpo.jld2") z
else
	println("Loading solution...")
    @load joinpath(@__DIR__, "sol_dpo.jld2") z
end
