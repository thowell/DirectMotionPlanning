# Model
include_model("acrobot")

# Horizon
T = 101

# Time step
tf = 5.0
h = tf / (T - 1)

# ul <= u <= uu
ul, uu = control_bounds(model, T, -10.0, 10.0)

# Initial and final states
x1 = [0.0; 0.0; 0.0; 0.0]
xT = [π; 0.0; 0.0; 0.0]
xl, xu = state_bounds(model, T, x1 = x1, xT = xT)

# Objective
obj = quadratic_tracking_objective(
        [t < T ? Diagonal(1.0 * ones(model.n)) : Diagonal(ones(model.n)) for t = 1:T],
        [Diagonal(1.0e-1 * ones(model.m)) for t = 1:T-1],
        [xT for t = 1:T],
        [zeros(model.m) for t = 1:T])

# Problem
prob = trajectory_optimization_problem(model,
           obj,
           T,
           h = h,
           xl = xl,
           xu = xu,
           ul = ul,
           uu = uu)

# Trajectory initialization
x0 = linear_interpolation(x1, xT, T) # linear interpolation on state
u0 = random_controls(model, T, 0.001) # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

# Solve
@time z, info = solve(prob, copy(z0), tol = 1.0e-6, c_tol = 1.0e-6)

# Visualize
using Plots
x, u = unpack(z, prob)
plot(hcat(x...)', width = 2.0)
plot(hcat(u...)', width = 2.0, linetype = :steppost)

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
visualize!(vis, model, x, Δt = h)

# Model-predictive control
include_snopt()

function run_mpc()
    T_mpc = 25
    x_ref = [x...,[x[end] for i = 1:T_mpc]...]
    u_ref = [u...,[u[end] for i = 1:T_mpc]...]
    Q_mpc = Diagonal(100.0 * ones(model.n))
    R_mpc = Diagonal(0.1 * ones(model.m))
    K, P = tvlqr(model, x, u, h, [Q_mpc for t = 1:T], [R_mpc for t = 1:T-1])
    P_mpc = [P..., [P[end] for i = 1:T_mpc]...]

    model_sim = Acrobot{RK3, FixedTime}(4, 1, 4,
        1.0 + 0.05, 0.33, 1.0, 0.5, 1.0 - 0.05, 0.33, 1.0, 0.5, 9.81, 0.1, 0.1)

    shift = 1

    x_hist = [x_ref[shift] + 1.0e-1 * randn(model.n)]
    u_hist = []

    for t = 1:T-1
    	println("T = $t")
    	ul_mpc, uu_mpc = control_bounds(model, T_mpc, -10.0, 10.0)
    	xl_mpc, xu_mpc = state_bounds(model, T_mpc, x1 = x_hist[end])

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
    			h = h,
    			ul = ul_mpc,
    			uu = uu_mpc,
    			xl = xl_mpc,
    			xu = xu_mpc)

    	# Pack trajectories into vector
    	z0_mpc = pack(x_ref[shift .+ (1:T_mpc)], u_ref[shift .+ (1:T_mpc)], prob_mpc)

    	# Solve
    	@time z_mpc, info = solve(prob_mpc, copy(z0_mpc),
            nlp = :SNOPT7, mapl = 0)
    	x_mpc, u_mpc = unpack(z_mpc, prob_mpc)

    	push!(x_hist,
    		fd(model_sim, x_hist[end], u_mpc[1], 1.0e-3 * randn(model_sim.d), h, nothing))
    	push!(u_hist, u_mpc[1])
    	shift += 1
    end
    return x_hist, u_hist
end

x_hist_mpc, u_hist_mpc = run_mpc()

plot(hcat(x...)',
	label = "", color = :black, width = 2.0)
plot!(hcat(x_hist_mpc...)',
	label = "", color = :magenta, width = 1.0)

plot(hcat(u...)',
	label = "u (ref)", color = :black, width = 2.0)
plot!(hcat(u_hist_mpc...)',
	label = "u (mpc)", color = :magenta, width = 1.0)

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
visualize!(vis, model, x_hist_mpc, Δt = h)
