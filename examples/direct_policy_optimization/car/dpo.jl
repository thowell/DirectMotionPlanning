include_dpo()
include(joinpath(@__DIR__, "car_obstacles.jl"))

# Additive noise model
model = additive_noise_model(model)

function fd(model::Model{Midpoint, FixedTime}, x⁺, x, u, w, h, t)
    x⁺ - (x + h * f(model, 0.5 * (x + x⁺), u, w) + w)
end

# Nominal solution
x̄, ū = unpack(z̄, prob)
prob_nom = prob.prob

# DPO
β = 1.0
# δ = 1.0e-3
δ = 1.0e-2

# Initial samples
x1_sample = resample(x1, Diagonal([1.0; 1.0; 0.1]), 1.0e-1)

# Mean problem
prob_mean = trajectory_optimization(
				model,
				EmptyObjective(),
				T,
				h = h,
				dynamics = false)

# Sample problems
prob_sample = [trajectory_optimization(
				model,
				EmptyObjective(),
				T,
				h = h,
				xl = state_bounds(model, T, x1 = x1_sample[i])[1],
				xu = state_bounds(model, T, x1 = x1_sample[i])[2],
				ul = ul,
				uu = uu,
				dynamics = false,
				con = con_obstacles) for i = 1:2 * model.n]

# Sample objective
Q = [(t < T ? Diagonal([10.0; 10.0; 1.0])
	: Diagonal(100.0 * ones(model.n))) for t = 1:T]
R = [Diagonal(1.0e-1 * ones(model.m)) for t = 1:T-1]

obj_sample = sample_objective(Q, R)
policy = linear_feedback(model.n, model.m)
dist = disturbances([Diagonal(δ * [1.0; 1.0; 0.1]) for t = 1:T-1])
sample = sample_params(β, T)

prob_dpo = dpo_problem(
	prob_nom, prob_mean, prob_sample,
	obj_sample,
	policy,
	dist,
	sample)

# TVLQR policy
K, P = tvlqr(model, x̄, ū, h, Q, R)

z0 = pack(z̄, K, prob_dpo)

# Solve
if false
	include_snopt()
	z, info = solve(prob_dpo, copy(z0),
		nlp = :SNOPT7,
		tol = 1.0e-3, c_tol = 1.0e-3,
		time_limit = 60 * 1)
	@save joinpath(@__DIR__, "sol_dpo.jld2") z
else
	println("Loading solution...")
	@load joinpath(@__DIR__, "sol_dpo.jld2") z
end
# include_snopt()
# z, info = solve(prob_dpo, copy(z0),
# 	nlp = :SNOPT7,
# 	tol = 1.0e-3, c_tol = 1.0e-3,
# 	time_limit = 60 * 5)
