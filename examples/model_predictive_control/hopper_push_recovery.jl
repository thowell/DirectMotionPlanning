# Model
include_model("hopper")
model_ft = free_time_model(model)

# Horizon
T = 51

# Time step
tf = 1.0
h = tf / (T - 1)

# Bounds
_uu = Inf * ones(model_ft.m)
_uu[model_ft.idx_u] = model_ft.uU
_uu[end] = 2.0 * h
_ul = zeros(model_ft.m)
_ul[model_ft.idx_u] = model_ft.uL
_ul[end] = 0.5 * h
ul, uu = control_bounds(model_ft, T, _ul, _uu)

# Initial and final states
z_h = 0.25
q1 = [0.0, 0.5 + z_h, 0.0, 0.25]

xl, xu = state_bounds(model_ft, T,
		[model_ft.qL; model_ft.qL],
		[model_ft.qU; model_ft.qU],
        x1 = [q1; Inf * ones(model.nq)])

# Objective
include_objective("velocity")
obj_tracking = quadratic_time_tracking_objective(
    [Diagonal(ones(model_ft.n)) for t = 1:T],
    [Diagonal([1.0, 1.0e-1,
		zeros(model_ft.nc)..., zeros(model_ft.nb)...,
		zeros(model_ft.m - model_ft.nu - model_ft.nc - model_ft.nb - 1)..., 0.0])
		for t = 1:T-1],
    [[q1; q1] for t = 1:T],
    [zeros(model_ft.m) for t = 1:T-1],
    1.0)

include_objective("control_velocity")
obj_contact_penalty = PenaltyObjective(1.0e5, model_ft.m - 1)
obj_velocity = velocity_objective(
    [1.0e-1 * Diagonal([1.0, 1.0, 1.0, 1.0]) for t = 1:T-1],
    model_ft.nq,
    h = 1.0,
    idx_angle = collect([3]))
obj_control_velocity = control_velocity_objective(Diagonal([1.0; 1.0; zeros(model_ft.m - 2)]))
obj = MultiObjective([obj_tracking,
	obj_contact_penalty,
	obj_velocity,
	obj_control_velocity])

# Constraints
include_constraints(["free_time", "contact", "loop"])
con_free_time = free_time_constraints(T)
con_contact = contact_constraints(model_ft, T)
con_loop = loop_constraints(model, (1:model.n), 1, T)
con = multiple_constraints([con_free_time, con_contact, con_loop])

# Problem
prob = trajectory_optimization_problem(model_ft,
               obj,
               T,
               xl = xl,
               xu = xu,
               ul = ul,
               uu = uu,
               con = con)

# Trajectory initialization
x0 = [[q1; q1] for t = 1:T] # linear interpolation on state
u0 = [[1.0e-5 * rand(model_ft.m-1); h] for t = 1:T-1] # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem

include_snopt()
@time z̄, info = solve(prob, copy(z0),
	nlp = :SNOPT7,
	tol = 1.0e-5, c_tol = 1.0e-5, mapl = 5,
	time_limit = 60)
@show check_slack(z̄, prob)
x̄, ū = unpack(z̄, prob)
tf, t, h̄ = get_time(ū)

z_traj = [x[2] for x in x̄]
λ_traj = [u[model_ft.idx_λ] for u in ū]
f_traj = [u[2] for u in ū]

using Plots
plot(h̄[1] * hcat(λ_traj...)', linetype = :steppost, label = ["" "λ1"])
plot!(h̄[1] * hcat(f_traj...)', linetype = :steppost, label = "force")

plot(hcat(z_traj...)', linetype = :steppost, label = "z")
plot(hcat(ū...)[1:model.nu, :]', linetype = :steppost, label = "u")

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
visualize!(vis, model_ft,
	# state_to_configuration(x̄),
	state_to_configuration([x̄..., ([x̄[2:end] for i = 1:3]...)...]),
	Δt = h̄[1],
	scenario = :vertical)

# Model-predictive control
include_snopt()
# include(joinpath(pwd(), "src/contact_simulator/simulator_variable_time_step.jl"))
include(joinpath(pwd(), "src/contact_simulator/simulator.jl"))

function run_mpc()
	# Time steps
	h_mpc = h̄[1] # mpc time step
	d_sim = 10 # simulator rate
	h_sim = h_mpc / d_sim # simulator time step

	# MPC horizon
	T_mpc =  3 * T # mpc horizon

	# Reference trajectories
	n_gaits = 20
	T_gait = n_gaits * T - (n_gaits - 1)
	_ū = [u[1:end-1] for u in ū] # remove time step control input
	x_ref = [x̄..., ([x̄[2:end] for i = 1:n_gaits-1]...)...]
	u_ref = [_ū..., ([_ū for i = 1:n_gaits-1]...)...]
	u_ref_ft = [ū..., ([ū for i = 1:n_gaits-1]...)...]

	Q_mpc = Diagonal([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
	R_mpc = Diagonal([1.0e-1 * ones(model.nu); 1.0e-1 * ones(model.m - model.nu)])
	R_mpc_ft = Diagonal([1.0e-1 * ones(model.nu); 1.0e-1 * ones(model.m - model.nu); 0.0])

	_, P_mpc = tvlqr(model_ft, x_ref, u_ref_ft, h_mpc,
		[Q_mpc for t = 1:T_gait], [R_mpc_ft for t = 1:T_gait-1])

	x1 = x_ref[1] + 1.0e-8 * randn(model.n)
    x_hist = [x1]
	q_sim = [x1[1:model.nq], x1[model.nq .+ (1:model.nq)]]
	v_sim = [(q_sim[end] - q_sim[end-1]) ./ h_mpc]
    u_hist = []

	T_sim = 400
	shift = 1

    for t = 1:T_sim-1

		# create MPC problem
    	println("T = $t")
		_uu = Inf * ones(model.m)
		_uu[model_ft.idx_u] = model.uU
		_ul = zeros(model.m)
		_ul[model.idx_u] = model.uL
		ul_mpc, uu_mpc = control_bounds(model, T_mpc, _ul, _uu)

		xl_mpc, xu_mpc = state_bounds(model, T_mpc,
				[model.qL; model.qL],
				[model.qU; model.qU],
		        x1 = x_hist[end])

    	# Objective
    	obj_mpc = quadratic_tracking_objective(
    	        [t < T_mpc ? Q_mpc : P_mpc[T_mpc + shift - 1] for t = 1:T_mpc],
				# [Q_mpc for t = 1:T_mpc],
    	        [R_mpc for t = 1:T_mpc-1],
    	        [x_ref[shift + t - 1] for t = 1:T_mpc],
    			[u_ref[shift + t - 1] for t = 1:T_mpc-1])

    	# Problem
    	prob_mpc = trajectory_optimization_problem(model,
    			obj_mpc,
    			T_mpc,
    			h = h_mpc,
    			ul = ul_mpc,
    			uu = uu_mpc,
    			xl = xl_mpc,
    			xu = xu_mpc)

    	# Pack trajectories into vector
    	z0_mpc = pack(x_ref[shift .+ (1:T_mpc)], u_ref[shift .+ (1:T_mpc-1)], prob_mpc)

    	# Solve MPC problem
    	@time z_mpc, info = solve(prob_mpc, copy(z0_mpc),
            nlp = :SNOPT7, mapl = 0, tol = 1.0e-3, c_tol = 1.0e-3)
    	x_mpc, u_mpc = unpack(z_mpc, prob_mpc)

		u_input = max.(min.(u_mpc[1], uu_mpc[1]), ul_mpc[1])[1:model.nu] # return feasible controls
		# u_input = u_mpc[1][1:model.nu]
		# simulate
		f_dist = 100.0
		w0 = t == T ? [f_dist, 0.0, 0.0, 0.0] : 0.0e-8 * randn(model.d) # disturbance
		# u_input = u_ref[shift][1:model.nu]
		# w0 = zeros(model.d)
		# push!(x_hist, step_contact(model, x_hist[end],
		# 		u_input, w0, h_mpc,
		# 		tol_c = 1.0e-5, tol_opt = 1.0e-5, tol_s = 1.0e-4,
		# 		nlp = :SNOPT7))

		for i = 1:d_sim
			# step simulator
			q = step_contact(model,
			   v_sim[end], q_sim[end-1], q_sim[end],
			   u_input, w0, h_sim)
			v = (q - q_sim[end]) / h_sim

			push!(q_sim, q)
			push!(v_sim, v)
		end

		push!(x_hist, [x_hist[end][model.nq .+ (1:model.nq)]; q_sim[end]])
    	push!(u_hist, u_input)
    	shift += 1
    end

    return x_hist, u_hist, x_ref[1:T_sim], u_ref[1:T_sim-1]
end

@time x_hist_mpc, u_hist_mpc, x_ref, u_ref = run_mpc()

plot(hcat(state_to_configuration(x_ref)...)',
	label = "", color = :black, width = 2.0)
plot!(hcat(state_to_configuration(x_hist_mpc)...)',
	label = "", color = :magenta, width = 1.0)

plot(hcat(u_ref...)[1:model.nu,:]',
	label = "u (ref)", color = :black, width = 2.0)
plot!(hcat(u_hist_mpc...)[1:model.nu,:]',
	label = "u (mpc)", color = :magenta, width = 1.0)

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
visualize!(vis, model, x_hist_mpc, Δt = h̄[1], scenario = :push)
