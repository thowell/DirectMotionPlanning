# Model
include_model("double_integrator")

function fd(model::DoubleIntegrator{Discrete, FixedTime}, x⁺, x, u, w, h, t)
    x⁺ - [x[1] + x[2] + w[1]; x[2] + (w[2] + 1.0) * u[1]]
end

function fd(model::DoubleIntegrator{Discrete, FixedTime}, x, u, w, h, t)
    [x[1] + x[2] + w[1]; x[2] + (w[2] + 1.0) * u[1]]
end

# Horizon
T = 101

# Time step
h = 1.0

# Initial and final states
x1 = [1.0; 0.0]
xT = [0.0; 0.0]

xl, xu = state_bounds(model, T, x1 = x1, xT = xT)
_ul = -0.01
_uu = 0.01
ul, uu = control_bounds(model, T, _ul, _uu)

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
               xu = xu,
			   ul = ul,
			   uu = uu)

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
	T_mpc = 25
	x_ref = [x...,[x[end] for i = 1:T_mpc]...]
	u_ref = [u...,[u[end] for i = 1:T_mpc]...]
	Q_mpc = Diagonal(10.0 * ones(model.n))
	R_mpc = Diagonal(1.0 * ones(model.m))
	K, P = tvlqr(model, x, u, h, [Q_mpc for t = 1:T], [R_mpc for t = 1:T-1])
	P_mpc = [P..., [P[end] for i = 1:T_mpc]...]

	shift = 1

	x_hist = [x_ref[shift] + 1.0e-1 * randn(model.n)]
	u_hist = []

	for t = 1:T-1
		println("T = $t")
		xl_mpc, xu_mpc = state_bounds(model, T_mpc, x1 = x_hist[end])
		ul_mpc, uu_mpc = control_bounds(model, T_mpc, _ul, _uu)
		# Objective
		obj_mpc = quadratic_tracking_objective(
		        [t < T_mpc ? Q_mpc : P_mpc[T_mpc + shift - 1] for t = 1:T_mpc],
		        [R_mpc for t = 1:T_mpc-1],
		        [x_ref[shift + t - 1] for t = 1:T_mpc],
				[u_ref[shift + t - 1] for t = 1:T_mpc-1])

		# Problem
		prob_mpc = trajectory_optimization_problem(model,
					   obj_mpc,
					   T_mpc,
		               xl = xl_mpc,
		               xu = xu_mpc,
					   ul = ul_mpc,
					   uu = uu_mpc)

		# Pack trajectories into vector
		z0_mpc = pack(x_ref[shift .+ (1:T_mpc)], u_ref[shift .+ (1:T_mpc-1)], prob_mpc)

		# Solve
		@time z_mpc, info = solve(prob_mpc, copy(z0_mpc))
		x_mpc, u_mpc = unpack(z_mpc, prob_mpc)

		# u_ctrl = -1.0 *  K[t] * (x_hist[end] - x_ref[t])
		# u_ctrl = u[t]
		u_ctrl = u_mpc[1]

		u_ctrl = min.(max.(_ul, u_ctrl[1]), _uu)

		push!(x_hist,
			fd(model, x_hist[end],
				u_ctrl,
				[0.0; 0.5] .+ 0.001 * randn(model.d), h, nothing))
		push!(u_hist, u_ctrl)
		shift += 1
	end

	return x_hist, u_hist
end

x_hist_mpc, u_hist_mpc = run_mpc()

plot(hcat(x...)',
	label = ["x (ref)" "v (ref)"], color = :black, width = 2.0)
plot!(hcat(x_hist_mpc...)',
	label = ["x (mpc)" "v (mpc)"], color = :magenta, width = 1.0)

plot(hcat(u...)',
	label = "u (ref)", color = :black, width = 2.0, linetype = :steppost)
plot!(hcat(u_hist_mpc...)',
	label = "u (mpc)", color = :magenta, width = 1.0, linetype = :steppost)
