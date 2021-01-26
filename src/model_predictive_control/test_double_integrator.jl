# Model
include_model("double_integrator")

# Horizon
T = 101

# Time step
h = 1.0

# Initial and final states
x1 = [1.0; 0.0]
xT = [0.0; 0.0]

xl, xu = state_bounds(model, T, x1 = x1, xT = xT)

# Objective
obj = quadratic_tracking_objective(
        [t < T ? Diagonal(ones(model.n)) : Diagonal(ones(model.n)) for t = 1:T],
        [Diagonal(100.0 * ones(model.m)) for t = 1:T-1],
        [xT for t = 1:T],
		[zeros(model.m) for t = 1:T])

# Problem
prob = trajectory_optimization_problem(model,
			   obj,
			   T,
               xl = xl,
               xu = xu)

# Initialization
x0 = linear_interpolation(x1, xT, T) # linear interpolation for states
u0 = [ones(model.m) for t = 1:T-1]

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

# Solve
@time z, info = solve(prob, copy(z0))

# Visualize
using Plots
x, u = unpack(z, prob)
plot(hcat(x...)', width = 2.0)
plot(hcat(u...)', width = 2.0, linetype = :steppost)

# Model-predictive control

function run_mpc()
	T_mpc = 40
	x_ref = [x...,[x[end] for i = 1:T_mpc]...]
	u_ref = [u...,[u[end] for i = 1:T_mpc]...]
	Q_mpc = Diagonal(ones(model.n))
	R_mpc = Diagonal(0.1 * ones(model.m))
	K, P = tvlqr(model, x, u, h, [Q_mpc for t = 1:T], [R_mpc for t = 1:T-1])
	P_mpc = [P..., [P[end] for i = 1:T_mpc]...]

	shift = 1

	x_hist = [x_ref[shift] + 1.0e-1 * randn(model.n)]
	u_hist = []

	for t = 1:T-1
		println("T = $T")
		xl_mpc, xu_mpc = state_bounds(model, T_mpc, x1 = x_hist[end])

		# Objective
		obj_mpc = quadratic_tracking_objective(
		        [t < T_mpc ? Q_mpc : P_mpc[T_mpc + shift - 1] for t = 1:T_mpc],
		        [R_mpc for t = 1:T_mpc-1],
		        [x_ref[shift + t - 1] for t = 1:T_mpc],
				[u_ref[shift + t - 1] for t = 1:T_mpc-1])

		# Problem
		prob_mpc = trajectory_optimization_problem(model,
					   obj,
					   T_mpc,
		               xl = xl_mpc,
		               xu = xu_mpc)

		# Pack trajectories into vector
		z0_mpc = pack(x_ref[shift .+ (1:T_mpc)], u_ref[shift .+ (1:T_mpc)], prob_mpc)

		# Solve
		@time z_mpc, info = solve(prob_mpc, copy(z0_mpc))
		x_mpc, u_mpc = unpack(z_mpc, prob_mpc)

		push!(x_hist,
			fd(model, x_hist[end], u_mpc[1], 0.01 * randn(model.d), h, nothing))
		push!(u_hist, u_mpc[end])
		shift += 1
	end
	return x_hist, u_hist
end

x_hist_mpc, u_hist_mpc = run_mpc()

plot(hcat(x...)',
	label = ["x (ref)" "v (ref)"], color = :black, width = 2.0)
plot!(hcat(x_hist_mpc...)',
	label = ["x (mpc)" "v (mpc)"], color = :orange, width = 1.0)
