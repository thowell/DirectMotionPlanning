# Model
include_model("walker")

# Visualize
# - Pkg.add any external deps from visualize.jl
include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)

"""
Useful methods

joint-space inertia matrix:
	M_func(model, q)

position (px, pz) of foot 1 heel:
	kinematics_2(model,
		get_q⁺(x), body = :calf_1, mode = :ee)

position (px, pz) of foot 2 heel:
	kinematics_2(model,
		get_q⁺(x), body = :calf_2, mode = :ee)

position (px, pz) of foot 1 toe:
		kinematics_3(model,
	        get_q⁺(x), body = :foot_1, mode = :ee)

position (px, pz) of foot 2 toe:
		kinematics_3(model,
	        get_q⁺(x), body = :foot_2, mode = :ee)

jacobian to foot 1 heel
	jacobian_2(model, get_q⁺(x), body = :calf_1, mode = :ee)

jacobian to foot 2 heel
	jacobian_2(model, get_q⁺(x), body = :calf_2, mode = :ee)

jacobian to foot 1 toe
	jacobian_3(model, get_q⁺(x), body = :foot_1, mode = :ee)

jacobian to foot 2 toe
	jacobian_3(model, get_q⁺(x), body = :foot_2, mode = :ee)

gravity compensating torques
	initial_torque(model, q, h)[model.idx_u] # returns torques for system
	initial_torque(model, q, h) # returns torques for system and contact forces

	-NOTE: this method is iterative and is not differentiable (yet)
		-TODO: implicit function theorem to compute derivatives

contact models
	- maximum-dissipation principle:
		contact_constraints(model, T)
	- no slip:
		contact_no_slip_constraints(T)

loop constraint
	- constrain variables at two indices to be the same:
		loop_constraints(model, x_idx, t1_idx, t2_idx)
"""

function get_q⁺(x)
	view(x, model.nq .+ (1:model.nq))
end

# Horizon
T = 51

# Time step
tf = 2.5
h = tf / (T - 1)

# Configurations
# 1: x pos
# 2: z pos
# 3: torso angle (rel. to downward vertical)
# 4: thigh 1 angle (rel. to downward vertical)
# 5: calf 1 (rel. to downward vertical)
# 6: thigh 2 (rel. to downward vertical)
# 7: calf 2 (rel. to downward vertical)
# 8: foot 1 (rel. to downward vertical)
# 9: foot 2 (rel. to downward vertical)

q1 = zeros(model.nq)
q1[8] = pi / 2.0
q1[9] = pi / 2.0
q1[2] = model.l_thigh1 + model.l_calf1

qT = copy(q1)
qT[1] = 1.0
q_ref = linear_interpolation(q1, qT, T)
visualize!(vis, model, q_ref, Δt = h)

# Control
# u = (τ1..7, λ1..4, β1..8, ψ1..4, η1...8, s1)
# τ1: torso angle
# τ2: thigh 1 angle
# τ3: calf 1
# τ4: thigh 2
# τ5: calf 2
# τ6: foot 1
# τ7: foot 2

# ul <= u <= uu
u1 = initial_torque(model, q1, h)[model.idx_u] # gravity compensation for current q
_uu = Inf * ones(model.m)
_uu[model.idx_u] .= 10.0
_ul = zeros(model.m)
_ul[model.idx_u] .= -10.0
ul, uu = control_bounds(model, T, _ul, _uu)

xl, xu = state_bounds(model, T,
    -Inf * ones(model.n), Inf * ones(model.n),
    x1 = [q1; q1], # initial state
    xT = [qT; qT]) # goal state

# Objective
include_objective(["velocity", "nonlinear_stage"])

x0 = configuration_to_state(q_ref)

# penalty on slack variable
obj_penalty = PenaltyObjective(1.0e5, model.m)

# quadratic tracking objective
# Σ (x - xref)' Q (x - x_ref) + (u - u_ref)' R (u - u_ref)
obj_control = quadratic_tracking_objective(
    [Diagonal(1.0e-3 * ones(model.n)) for t = 1:T],
    [Diagonal([1.0e-1 * ones(model.nu)..., 0.0 * ones(model.m - model.nu)...]) for t = 1:T-1],
    [x0[end] for t = 1:T],
    [[u1; zeros(model.m - model.nu)] for t = 1:T-1])

# quadratic velocity penalty
# Σ v' Q v
q_v = 1.0e-1 * ones(model.nq)
obj_velocity = velocity_objective(
    [Diagonal(q_v) for t = 1:T-1],
    model.nq,
    h = h,
    idx_angle = collect([3, 4, 5, 6, 7, 8 ,9]))

# torso height
t_h = kinematics_1(model, q1, body = :torso, mode = :com)[2]
function l_stage_torso_h(x, u, t)
    return 10.0 * (kinematics_1(model,
            get_q⁺(x), body = :torso, mode = :com)[2] - t_h)^2.0
end
l_terminal_torso_h(x) = 0.0
obj_th = nonlinear_stage_objective(l_stage_torso_h, l_terminal_torso_h)

# foot 1 height
function l_stage_fh1(x, u, t)
    return 10.0 * (kinematics_3(model,
        get_q⁺(x), body = :foot_1, mode = :com)[2] - 0.1)^2.0
end
l_terminal_fh1(x) = 0.0
obj_fh1 = nonlinear_stage_objective(l_stage_fh1, l_terminal_fh1)

# foot 2 height
function l_stage_fh2(x, u, t)
    return 10.0 * (kinematics_3(model,
        get_q⁺(x), body = :foot_2, mode = :com)[2] - 0.1)^2.0
end
l_terminal_fh2(x) = 0.0
obj_fh2 = nonlinear_stage_objective(l_stage_fh2, l_terminal_fh2)

# (torso lateral - feet average)^2
function l_stage_torso_feet(x, u, t)
	x_torso = kinematics_1(model,
        get_q⁺(x), body = :torso, mode = :com)[1]
	x_foot1 = kinematics_3(model,
		        get_q⁺(x), body = :foot_1, mode = :com)[1]
	x_foot2 = kinematics_3(model,
				        get_q⁺(x), body = :foot_2, mode = :com)[1]

	return 10.0 * (x_torso - (x_foot2 - x_foot1) / 2.0)^2.0
end
l_terminal_torso_feet(x) = 0.0
obj_tf = nonlinear_stage_objective(l_stage_torso_feet, l_terminal_torso_feet)

function l_stage_forward(x, u, t)
	px = kinematics_1(model,
            get_q⁺(x), body = :torso, mode = :com)[1]

	return 1.0 * (px - qT[1])^2.0
end
l_terminal_forward(x) = 0.0
obj_forward = nonlinear_stage_objective(l_stage_forward, l_terminal_forward)

obj = MultiObjective([obj_penalty,
                      obj_control,
                      obj_velocity,
                      obj_th,
                      obj_tl,
                      obj_fh1,
                      obj_fh2,
					  obj_tf])
					  # obj_forward])

# Constraints
include_constraints(["contact", "loop", "contact_no_slip", "free_time"])
# con_loop = loop_constraints(model, collect([(2:7)...,(9:14)...]), 1, T)
con_contact = contact_constraints(model, T)
# con_free_time = free_time_constraints(T)
con = multiple_constraints([con_contact])#, con_free_time, con_loop])

# Problem
prob = trajectory_optimization_problem(model,
               obj,
               T,
               h = h,
               xl = xl,
               xu = xu,
               ul = ul,
               uu = uu,
               con = con)

# trajectory initialization
u0 = [[u1; 1.0e-3 * rand(model.m - model.nu)] for t = 1:T-1] # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob) + 0.005 * rand(prob.num_var)

# Solve
include_snopt()
@time z̄, info = solve(prob, copy(z0),
    nlp = :SNOPT7,
    tol = 1.0e-3, c_tol = 1.0e-3, mapl = 5,
    time_limit = 60 * 3)
@show check_slack(z̄, prob)
x̄, ū = unpack(z̄, prob)
visualize!(vis, model, state_to_configuration(x̄), Δt = h)

# if false
#     include_snopt()
# 	@time z̄ , info = solve(prob, copy(z0),
# 		nlp = :SNOPT7,
# 		tol = 1.0e-3, c_tol = 1.0e-3, mapl = 5,
# 		time_limit = 60 * 3)
# 	@show check_slack(z̄, prob)
# 	x̄, ū = unpack(z̄, prob)
#     tfc, tc, h̄ = get_time(ū)
#
# 	#projection
# 	Q = [Diagonal(ones(model.n)) for t = 1:T]
# 	R = [Diagonal(0.1 * ones(model.m)) for t = 1:T-1]
# 	x_proj, u_proj = lqr_projection(model, x̄, ū, h̄[1], Q, R)
#
# 	@show tfc
# 	@show h̄[1]
# 	@save joinpath(pwd(), "examples/trajectories/walker_steps.jld2") x̄ ū h̄ x_proj u_proj
# else
# 	@load joinpath(pwd(), "examples/trajectories/walker_steps.jld2") x̄ ū h̄ x_proj u_proj
# end

# Visualize
vis = Visualizer()
render(vis)
visualize!(vis, model, state_to_configuration(x̄), Δt = h)

using Plots
fh1 = [kinematics_2(model,
    state_to_configuration(x̄)[t], body = :calf_1, mode = :ee)[2] for t = 1:T]
fh2 = [kinematics_2(model,
    state_to_configuration(x̄)[t], body = :calf_2, mode = :ee)[2] for t = 1:T]
plot(fh1, linetype = :steppost, label = "foot 1")
plot!(fh2, linetype = :steppost, label = "foot 2")

plot(hcat(ū...)[1:7, :]',
    linetype = :steppost,
    label = "",
    color = :red,
    width = 2.0)
#
# plot!(hcat(u_proj...)[1:4, :]',
#     linetype = :steppost,
#     label = "",
#     color = :black)
#
# plot(hcat(u_proj...)[5:6, :]',
#     linetype = :steppost,
#     label = "",
#     width = 2.0)

plot(hcat(state_to_configuration(x̄)...)'[1:3],
    color = :red,
    width = 2.0,
    label = "")
#
# plot!(hcat(state_to_configuration(x_proj)...)',
#     color = :black,
#     label = "")
