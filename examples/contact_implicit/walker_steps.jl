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
Tm = convert(Int, floor(0.5 * T))

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
    # q1[6] = θ_thigh_2
    # q1[7] = -1.0 * acos((z1 - model.l_thigh2 * cos(q1[6])) / model.l_calf2)
	q1[6] = θ_thigh_1
	q1[7] = θ_leg_1
	q1[2] = z1
	q1[8] = pi / 2.0
	q1[9] = pi / 2.0

    p1 = kinematics_2(model, q1, body = :calf_1, mode = :ee)
    p2 = kinematics_2(model, q1, body = :calf_2, mode = :ee)
    @show stride = abs(p1[1] - p2[1])

    q1[1] = -0.75#-1.0 * p1[1]

    qM = copy(q1)
    qM[4] = q1[6]
    qM[5] = q1[7]
    qM[6] = q1[4]
    qM[7] = q1[5]
    qM[1] = abs(p2[1])

    pM_1 = kinematics_2(model, qM, body = :calf_1, mode = :ee)
    pM_2 = kinematics_2(model, qM, body = :calf_2, mode = :ee)

    qT = copy(q1)
    qT[1] = 0.75#q1[1] + 1.0#2 * stride

    pT_1 = kinematics_2(model, qT, body = :calf_1, mode = :ee)
    pT_2 = kinematics_2(model, qT, body = :calf_2, mode = :ee)

    return q1, qM, qT
end

q1, qM, qT = initial_configuration(model, -pi / 100.0, pi / 20.0, -pi / 10.0, -pi / 25.0)
q_ref = linear_interpolation(q1, qT, T)
x_ref = configuration_to_state(q_ref)
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

qL = [-Inf; -Inf; q1[3:end] .- pi / 4.0; -Inf; -Inf; q1[3:end] .- pi / 4.0]
qU = [Inf; q1[2] + 0.001; q1[3:end] .+ pi / 4.0; Inf; Inf; q1[3:end] .+ pi / 4.0]

q2 = copy(q1)
q2[1] = q1[1] + 0.25 * h
xl, xu = state_bounds(model, T,
    qL, qU,
    x1 = [q1; q2], # initial state
    xT = [Inf * ones(model.nq); qT]) # goal state

# Objective
include_objective(["velocity", "nonlinear_stage", "control_velocity"])

x0 = configuration_to_state(q_ref)

# penalty on slack variable
obj_penalty = PenaltyObjective(1.0e4, model.m)

# quadratic tracking objective
# Σ (x - xref)' Q (x - x_ref) + (u - u_ref)' R (u - u_ref)
q_penalty = 0.1 * ones(model.nq)
q_penalty[1] = 0.0
q_penalty[2] = 10.0
q_penalty[3] = 1.0
x_penalty = [q_penalty; q_penalty]
obj_control = quadratic_tracking_objective(
    [h * Diagonal(x_penalty) for t = 1:T],
    [h * Diagonal([1.0 * ones(model.nu)..., 0.0 * ones(model.m - model.nu)...]) for t = 1:T-1],
    [x_ref[T] for t = 1:T],
    [[zeros(model.nu); zeros(model.m - model.nu)] for t = 1:T-1])

# quadratic velocity penalty
# Σ v' Q v
q_v = 10.0 * ones(model.nq)
# q_v[1] = 0.0
# q_v[2] = 0.0
obj_velocity = velocity_objective(
    [h * Diagonal(q_v) for t = 1:T-1],
    model.nq,
    h = h)#,
    # idx_angle = collect([3, 4, 5, 6, 7, 8 ,9]))

function l_foot_height(x, u, t)
	ϕ_des = [0.1; 0.1; 0.1; 0.1]
	ϕ = ϕ_func(model, get_q⁺(x))

	idx = (ϕ .< ϕ_des)

	return 1000.0 * (ϕ[idx] - ϕ_des[idx])' * (ϕ[idx] - ϕ_des[idx])
end

l_foot_height(x) = l_foot_height(x, nothing, T)

obj_foot_height = nonlinear_stage_objective(l_foot_height, l_foot_height)

obj = MultiObjective([obj_penalty,
                      obj_control,
                      obj_velocity,
					  obj_foot_height])
					  # obj_control_velocity])
                      # obj_th])#,
                      # obj_fh1,
                      # obj_fh2,
					  # obj_tf])
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
u0 = [0.1 * randn(model.m) for t = 1:T-1] # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob)
# Solve
include_snopt()
@time z̄, info = solve(prob, copy(z0),
    nlp = :SNOPT7,
    tol = 1.0e-4, c_tol = 1.0e-3, mapl = 5,
    time_limit = 60 * 1)

@show check_slack(z̄, prob)
x̄, ū = unpack(z̄, prob)
vis = Visualizer()
render(vis)
visualize!(vis, model,
	[[x̄[1][1:model.nq] for i = 1:10]...,
	 state_to_configuration(x̄)...,
	 [x̄[T][model.nq .+ (1:model.nq)] for i = 1:10]...], Δt = h)
open(vis)
# @save joinpath(pwd(), "examples/trajectories/walker_steps.jld2") z̄

# using Plots
# fh1 = [kinematics_3(model,
#     state_to_configuration(x̄)[t], body = :foot_1, mode = :com)[2] for t = 1:T]
# fh2 = [kinematics_3(model,
#     state_to_configuration(x̄)[t], body = :foot_2, mode = :com)[2] for t = 1:T]
# plot(fh1, linetype = :steppost, label = "foot 1")
# plot!(fh2, linetype = :steppost, label = "foot 2")

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
