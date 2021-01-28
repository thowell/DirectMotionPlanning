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
T = 26

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
function initial_configuration(model, θ_torso, θ_thigh_1, θ_leg_1, θ_thigh_2)
    q1 = zeros(model.nq)
    q1[3] = θ_torso
    q1[4] = θ_thigh_1
    q1[5] = θ_leg_1
    z1 = model.l_thigh1 * cos(q1[4]) + model.l_calf1 * cos(q1[5])
    q1[6] = θ_thigh_2
    q1[7] = -1.0 * acos((z1 - model.l_thigh2 * cos(q1[6])) / model.l_calf2)
    q1[2] = z1
	q1[8] = pi / 2.0
	q1[9] = pi / 2.0

    p1 = kinematics_2(model, q1, body = :calf_1, mode = :ee)
    p2 = kinematics_2(model, q1, body = :calf_2, mode = :ee)
    @show stride = abs(p1[1] - p2[1])

    q1[1] = -1.0 * p1[1]

    qM = copy(q1)
    qM[4] = q1[6]
    qM[5] = q1[7]
    qM[6] = q1[4]
    qM[7] = q1[5]
    qM[1] = abs(p2[1])

    pM_1 = kinematics_2(model, qM, body = :calf_1, mode = :ee)
    pM_2 = kinematics_2(model, qM, body = :calf_2, mode = :ee)

    qT = copy(q1)
    qT[1] = 2 * stride

    pT_1 = kinematics_2(model, qT, body = :calf_1, mode = :ee)
    pT_2 = kinematics_2(model, qT, body = :calf_2, mode = :ee)

    return q1, qM, qT
end
q1, qM, qT = initial_configuration(model, -pi / 100.0, pi / 10.0, -pi / 25.0, -pi / 25.0)
q_ref = [linear_interpolation(q1, qM, Tm)...,
    linear_interpolation(qM, qT, Tm)[1:end]...]
# q_ref = linear_interpolation(q1, qT, T)
x_ref = configuration_to_state(q_ref)
visualize!(vis, model, q_ref, Δt = h)
visualize!(vis, model, [q1], Δt = h)

ϕ_func(model, q1)

# feet positions
Tm = convert(Int64, floor(0.5 * T))

foot_2_1 = kinematics_2(model, q1, body = :calf_2, mode = :ee)
foot_2_M = kinematics_2(model, qM, body = :calf_2, mode = :ee)

foot_1_M = kinematics_2(model, qM, body = :calf_1, mode = :ee)
foot_1_T = kinematics_2(model, qT, body = :calf_1, mode = :ee)

# feet trajectories
zh = 0.1
foot_2_x = [range(foot_2_1[1], stop = foot_2_M[1], length = Tm)...,
    [foot_2_M[1] for t = 1:Tm]...]
foot_2_z = sqrt.((zh^2.0) * (1.0 .- ((foot_2_x).^2.0)
    ./ abs(foot_2_1[1])^2.0) .+ 1.0e-8)

foot_1_x = [[foot_1_M[1] for t = 1:Tm]...,
    range(foot_1_M[1], stop = foot_1_T[1], length = Tm)...]
foot_1_z = sqrt.((zh^2.0) * (1.0 .- ((foot_1_x .- abs(foot_1_T[1] / 2.0)).^2.0)
    ./ abs(foot_1_T[1] / 2.0)^2.0) .+ 1.0e-8)

using Plots
plot(foot_2_x, foot_2_z)
plot!(foot_1_x, foot_1_z, aspect_ratio = :equal)

t = range(0.0, stop = tf, length = T)
length(t)
length(foot_2_x)
plot(t, foot_2_x)
plot!(t, foot_1_x)

plot(t, foot_2_z)
plot!(t, foot_1_z)

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
_uu[model.idx_u] .= 5.0
_ul = zeros(model.m)
_ul[model.idx_u] .= -5.0
ul, uu = control_bounds(model, T, _ul, _uu)

qL = [-Inf; -Inf; q1[3:end] .- pi / 4.0; -Inf; -Inf; q1[3:end] .- pi / 4.0]
qU = [Inf; Inf; q1[3:end] .+ pi / 4.0; Inf; Inf; q1[3:end] .+ pi / 4.0]

xl, xu = state_bounds(model, T,
    qL, qU,
    x1 = [q1; q1], # initial state
    xT = [qT; qT]) # goal state

# Objective
include_objective(["velocity", "nonlinear_stage", "control_velocity"])

x0 = configuration_to_state(q_ref)

# penalty on slack variable
obj_penalty = PenaltyObjective(1.0e3, model.m)

# quadratic tracking objective
# Σ (x - xref)' Q (x - x_ref) + (u - u_ref)' R (u - u_ref)
obj_control = quadratic_tracking_objective(
    [Diagonal(0.0 * ones(model.n)) for t = 1:T],
    [Diagonal([0.0 * ones(model.nu)..., 0.0 * ones(model.m - model.nu)...]) for t = 1:T-1],
    [x_ref[t] for t = 1:T],
    [[u1; zeros(model.m - model.nu)] for t = 1:T-1])

# quadratic velocity penalty
# Σ v' Q v
q_v = 1.0e-2 * ones(model.nq)
q_v[2] = 1.0e-1
q_v[3] = 1.0
obj_velocity = velocity_objective(
    [Diagonal(q_v) for t = 1:T-1],
    model.nq,
    h = h,
    idx_angle = collect([3, 4, 5, 6, 7, 8 ,9]))

obj_control_velocity = control_velocity_objective(Diagonal([1.0e-1 * ones(model.nu); 0.0 * ones(model.m - model.nu)]))

# torso height
t_h = kinematics_1(model, q1, body = :torso, mode = :com)[2]
function l_stage_torso_h(x, u, t)
    return 1.0 * (kinematics_1(model,
            get_q⁺(x), body = :torso, mode = :com)[2] - t_h)^2.0
end
l_terminal_torso_h(x) = 0.0
obj_th = nonlinear_stage_objective(l_stage_torso_h, l_terminal_torso_h)

# foot 1 height
function l_stage_fh1(x, u, t)
    return 50.0 * (kinematics_3(model,
        get_q⁺(x), body = :foot_1, mode = :com)[2] - foot_1_z[t])^2.0
end
l_terminal_fh1(x) = 0.0
obj_fh1 = nonlinear_stage_objective(l_stage_fh1, l_terminal_fh1)

# foot 2 height
function l_stage_fh2(x, u, t)
    return 50.0 * (kinematics_3(model,
        get_q⁺(x), body = :foot_2, mode = :com)[2] - foot_2_z[t])^2.0
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

	return 0.0 * (x_torso - (x_foot2 - x_foot1) / 2.0)^2.0
end
l_terminal_torso_feet(x) = 0.0
obj_tf = nonlinear_stage_objective(l_stage_torso_feet, l_terminal_torso_feet)

# function l_stage_forward(x, u, t)
# 	px = kinematics_1(model,
#             get_q⁺(x), body = :torso, mode = :com)[1]
#
# 	return 1.0 * (px - qT[1])^2.0
# end
# l_terminal_forward(x) = 0.0
# obj_forward = nonlinear_stage_objective(l_stage_forward, l_terminal_forward)

obj = MultiObjective([obj_penalty,
                      obj_control,
                      obj_velocity,
					  obj_control_velocity,
                      obj_th,
                      obj_fh1,
                      obj_fh2,
					  obj_tf])
					  # obj_forward])

# Constraints
include_constraints(["contact"])#, "loop", "free_time"])
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
u0 = [[u1; 0.0e-8 * rand(model.m - model.nu)] for t = 1:T-1] # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob) + 0.0e-8 * rand(prob.num_var)

# Solve
include_snopt()
@time z̄, info = solve(prob, copy(z0),
    nlp = :SNOPT7,
    tol = 1.0e-3, c_tol = 1.0e-3, mapl = 5,
    time_limit = 60 * 3)
@show check_slack(z̄, prob)
x̄, ū = unpack(z̄, prob)
visualize!(vis, model, state_to_configuration(x̄), Δt = h)

# using Plots
# fh1 = [kinematics_2(model,
#     state_to_configuration(x̄)[t], body = :calf_1, mode = :ee)[2] for t = 1:T]
# fh2 = [kinematics_2(model,
#     state_to_configuration(x̄)[t], body = :calf_2, mode = :ee)[2] for t = 1:T]
# plot(fh1, linetype = :steppost, label = "foot 1")
# plot!(fh2, linetype = :steppost, label = "foot 2")
#
# plot(hcat(ū...)[1:7, :]',
#     linetype = :steppost,
#     label = "",
#     color = :red,
#     width = 2.0)
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

# plot(hcat(state_to_configuration(x̄)...)'[1:3],
#     color = :red,
#     width = 2.0,
#     label = "")
#
# plot!(hcat(state_to_configuration(x_proj)...)',
#     color = :black,
#     label = "")
