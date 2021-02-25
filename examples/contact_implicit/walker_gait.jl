using Plots

# Model
include_model("walker")

model = free_time_model(model)

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

function ellipse_traj(x_start, x_goal, z, T)
	dist = x_goal - x_start
	a = 0.5 * dist
	b = z

	z̄ = 0.0

	x = range(x_start, stop = x_goal, length = T)

	z = sqrt.(max.(0.0, (b^2) * (1.0 .- ((x .- (x_start + a)).^2.0) / (a^2.0))))

	return x, z
end

# Horizon
T = 50
Tm = 26 #convert(Int, floor(0.5 * T))

# Time step
tf = 0.5
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
	# q1[6] = θ_thigh_1
	# q1[7] = θ_leg_1
	q1[2] = z1
	q1[8] = pi / 2.0
	q1[9] = pi / 2.0

    # p1 = kinematics_2(model, q1, body = :calf_1, mode = :ee)
    # p2 = kinematics_2(model, q1, body = :calf_2, mode = :ee)
    # @show stride = abs(p1[1] - p2[1])
	#
    # q1[1] = -0.75#-1.0 * p1[1]
	#
    # qM = copy(q1)
    # qM[4] = q1[6]
    # qM[5] = q1[7]
    # qM[6] = q1[4]
    # qM[7] = q1[5]
    # qM[1] = abs(p2[1])
	#
    # pM_1 = kinematics_2(model, qM, body = :calf_1, mode = :ee)
    # pM_2 = kinematics_2(model, qM, body = :calf_2, mode = :ee)
	#
    # qT = copy(q1)
    # qT[1] = 0.75#q1[1] + 1.0#2 * stride
	#
    # pT_1 = kinematics_2(model, qT, body = :calf_1, mode = :ee)
    # pT_2 = kinematics_2(model, qT, body = :calf_2, mode = :ee)

    return q1
end

q1 = initial_configuration(model, -pi / 100.0, pi / 20.0, pi / 50.0, -pi / 40.0)

pf1 = kinematics_3(model, q1, body = :foot_1, mode = :com)
pf2 = kinematics_3(model, q1, body = :foot_2, mode = :com)

# pt1 = kinematics_3(model, q1, body = :foot_1, mode = :toe)
# ph1 = kinematics_3(model, q1, body = :foot_1, mode = :heel)
#
# pt2 = kinematics_3(model, q1, body = :foot_2, mode = :toe)
# ph2 = kinematics_3(model, q1, body = :foot_2, mode = :heel)

strd = 2 * (pf1 - pf2)[1]

zh = 0.075
xf2_el, zf2_el = ellipse_traj(pf2[1], pf2[1] + strd, zh, Tm-1)
xf1_el, zf1_el = ellipse_traj(pf1[1], pf1[1] + strd, zh, Tm-1)

zf2 = [zf2_el..., [zf2_el[end] for t = 1:Tm-1]...]
xf2 = [xf2_el..., [xf2_el[end] for t = 1:Tm-1]...]
zf1 = [[zf1_el[1] for t = 1:Tm-1]..., zf1_el...]
xf1 = [[xf1_el[1] for t = 1:Tm-1]..., xf1_el...]

p1_ref = [[xf1[t]; zf1[t]] for t = 1:T]
p2_ref = [[xf2[t]; zf2[t]] for t = 1:T]

plot(hcat(p1_ref...)', legend = :topleft)
plot(hcat(p2_ref...)')

qT = copy(q1)
qT[1] += strd
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
# u1 = initial_torque(model, q1, h)[model.idx_u] # gravity compensation for current q
_uu = Inf * ones(model.m)
_uu[model.idx_u] .= 25.0
_uu[end] = 2.0 * h
_ul = zeros(model.m)
_ul[model.idx_u] .= -25.0
_ul[end] = 0.25 * h
ul, uu = control_bounds(model, T, _ul, _uu)

qL = [-Inf; -Inf; q1[3] - pi / 50.0; q1[4:end] .- pi / 6.0; -Inf; -Inf; q1[3] - pi / 50.0; q1[4:end] .- pi / 6.0]
qU = [Inf; q1[2] + 0.001; 0.0; q1[4:end] .+ pi / 6.0; Inf; Inf; q1[3:end] .+ pi / 6.0]
qL[8] = q1[8] - pi / 10.0
qL[9] = q1[9] - pi / 10.0
qL[9 + 8] = q1[8] - pi / 10.0
qL[9 + 9] = q1[9] - pi / 10.0

qU[8] = q1[8] + pi / 10.0
qU[9] = q1[9] + pi / 10.0
qU[9 + 8] = q1[8] + pi / 10.0
qU[9 + 9] = q1[9] + pi / 10.0

xl, xu = state_bounds(model, T,
    qL, qU,
    x1 = [Inf * ones(model.nq); q1],
	xT = [Inf * ones(model.nq); qT[1]; Inf * ones(model.nq - 1)])

# Objective
include_objective(["velocity", "nonlinear_stage", "control_velocity"])

x0 = configuration_to_state(q_ref)

# penalty on slack variable
obj_penalty = PenaltyObjective(1.0e4, model.m-1)

# quadratic tracking objective
# Σ (x - xref)' Q (x - x_ref) + (u - u_ref)' R (u - u_ref)
q_penalty = 1.0e-5 * ones(model.nq)
# q_penalty[1] = 1.0
# q_penalty[2] = 1.0
# q_penalty[3] = 10.0
# q_penalty[8] = 10.0
# q_penalty[9] = 10.0
x_penalty = [q_penalty; q_penalty]
obj_control = quadratic_time_tracking_objective(
    [Diagonal(x_penalty) for t = 1:T],
    [Diagonal([1.0e-5 * ones(model.nu)..., 1.0e-8 * ones(model.m - model.nu)...]) for t = 1:T-1],
    [x_ref[end] for t = 1:T],
    [[zeros(model.nu); zeros(model.m - model.nu)] for t = 1:T-1],
	1.0)

obj_ctrl_vel = control_velocity_objective(Diagonal([1.0e-3 * ones(model.nu); zeros(model.m - model.nu)]))

# quadratic velocity penalty
q_v = 1.0e-1 * ones(model.nq)
# q_v[3] = 100.0
# q_v[3:7] .= 1.0e-3
# q_v[8:9] .= 100.0
obj_velocity = velocity_objective(
    [Diagonal(q_v) for t = 1:T-1],
    model.nq,
    h = h)

function l_foot_height(x, u, t)
	J = 0.0
	q1 = view(x, 1:9)
	q2 = view(x, 9 .+ (1:9))

	if t >= Tm
		pq1 = kinematics_3(model, q1, body = :foot_1, mode = :com)
		pq2 = kinematics_3(model, q1, body = :foot_1, mode = :com)
		v = (pq2 - pq1) ./ h
		J += 100000.0 * sum((p1_ref[t] - kinematics_3(model, q1, body = :foot_1, mode = :toe)).^2.0)
		J += 100000.0 * sum((p1_ref[t] - kinematics_3(model, q1, body = :foot_1, mode = :heel)).^2.0)
		# J += 1000.0 * v' * v
	end

	if t <= Tm
		pq1 = kinematics_3(model, q1, body = :foot_2, mode = :com)
		pq2 = kinematics_3(model, q2, body = :foot_2, mode = :com)
		v = (pq2 - pq1) ./ h
		J += 100000.0 * sum((p2_ref[t] - kinematics_3(model, q1, body = :foot_2, mode = :toe)).^2.0)
		J += 100000.0 * sum((p2_ref[t] - kinematics_3(model, q2, body = :foot_2, mode = :heel)).^2.0)
		# J += 1000.0 * v' * v
	end

	if t < T
		J += 1.0e-3 * ((q2[1] - q1[1]) / max(1.0e-6, u[model.m]) - 1.0)^2.0
	end

	return J
end

l_foot_height(x) = l_foot_height(x, nothing, T)

obj_foot_height = nonlinear_stage_objective(l_foot_height, l_foot_height)

obj = MultiObjective([obj_penalty,
                      obj_control,
                      obj_velocity,
					  obj_ctrl_vel,
					  obj_foot_height])
# Constraints
include_constraints(["contact", "loop", "free_time", "stage"])

function pinned1!(c, x, u, t)
    q1 = view(x, 1:9)
	# q2 = view(x, 9 .+ (1:9))
    c[1:2] = p1_ref[t] - kinematics_3(model, q1, body = :foot_1, mode = :com)
	# c[3:4] = p2_ref[t] - kinematics_3(model, q1, body = :foot_2, mode = :com)
	nothing
end

function pinned2!(c, x, u, t)
    q1 = view(x, 1:9)
    # c[1:2] = p1_ref[t] - kinematics_3(model, q, body = :foot_1, mode = :com)
	c[1:2] = p2_ref[t] - kinematics_3(model, q1, body = :foot_2, mode = :com)
	nothing
end

# cc = zeros(4)
# xx = zeros(model.n)
# uuu = zeros(model.m)
# pinned!(cc, xx, uuu, 1)
# p1_ref[1]
n_stage = 2
t_idx1 = [t for t = 1:Tm]
t_idx2 = [t for t = Tm:T-1]

con_pinned1 = stage_constraints(pinned1!, n_stage, (1:0), t_idx1)
con_pinned2 = stage_constraints(pinned2!, n_stage, (1:0), t_idx2)

# constraints!(zeros(con_pinned.n), zeros(prob.num_var), con_pinned, model, prob.prob.idx, h, T)

con_loop = loop_constraints(model, collect([(2:9)...,(11:18)...]), 1, T)
con_contact = contact_constraints(model, T)
con_free_time = free_time_constraints(T)
con = multiple_constraints([con_contact,
	con_free_time,
	con_loop,
	con_pinned1,
	con_pinned2])#,
	# con_pinned2])#,
	# con_pinned2])#,
	# con_pinned1,
	# con_pinned2])

# con = multiple_constraints([con_free_time, con_contact, con_pinned1])

# Problem
prob = trajectory_optimization_problem(model,
               obj,
               T,
               xl = xl,
               xu = xu,
               ul = ul,
               uu = uu,
               con = con)
qT[1]
# trajectory initialization
u0 = [[1.0e-2 * randn(model.nu); 0.01 * randn(model.m - model.nu - 1); h] for t = 1:T-1] # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

# Solve
# NOTE: run multiple times to get good trajectory
include_snopt()
@time z̄, info = solve(prob, copy(z0),
    nlp = :ipopt,
	max_iter = 1000,
    tol = 1.0e-1, c_tol = 1.0e-3, mapl = 5,
    time_limit = 60 * 3)

# @time z̄, info = solve(prob, copy(z̄ .+ 0.01 * randn(prob.num_var)),
#     nlp = :SNOPT7,
# 	max_iter = 1000,
#     tol = 1.0e-3, c_tol = 1.0e-3, mapl = 5,
#     time_limit = 60 * 3)

@show check_slack(z̄, prob)
x̄, ū = unpack(z̄, prob)
q̄ = state_to_configuration(x̄)
τ̄ = [u[model.idx_u] for u in ū]
λ̄ = [u[model.idx_λ] for u in ū]
b̄ = [u[model.idx_b] for u in ū]
tf_, t_, h̄ = get_time(ū)

# @save joinpath(@__DIR__, "walker_gait.jld2") z̄ x̄ ū q̄ τ̄ λ̄ b̄ h̄
# @load joinpath(@__DIR__, "walker_gait.jld2") z̄ x̄ ū q̄ τ̄ λ̄ b̄ h̄

[norm(fd(model, x̄[t+1], x̄[t], ū[t], zeros(model.d), h̄[t], t)) for t = 1:T-1]
(q̄[end][1] - q̄[1][1]) / tf_
vis = Visualizer()
render(vis)
visualize!(vis, model,
	[[x̄[1][1:model.nq] for i = 1:10]...,
	 state_to_configuration(x̄)...,
	 [x̄[end][model.nq .+ (1:model.nq)] for i = 1:10]...], Δt = h̄[1])
# visualize!(vis, model,
#  	[q̄[T+1]], Δt = h̄[1])

_pf1 = [kinematics_3(model, q, body = :foot_1, mode = :com) for q in q̄]
_pf2 = [kinematics_3(model, q, body = :foot_2, mode = :com) for q in q̄]

using Plots
plot(hcat(_pf1...)', title = "foot 1", legend = :topleft, label = ["x" "z"])
plot(hcat(_pf2...)', title = "foot 2", legend = :bottomright, label = ["x" "z"])

# # @save joinpath(pwd(), "examples/trajectories/walker_steps.jld2") z̄
# plot(hcat(ū...)[1:model.nu, :]', linetype = :steppost)
function get_q_viz(q̄)
	q_viz = [q̄...]
	shift_vec = zeros(model.nq)
	shift_vec[1] = q̄[end][1]
	for i = 1:3
		# println(shift)
		# shift_vec[1] = strd
		#
		q_update = [q + shift_vec for q in q̄[2:end]]
		push!(q_viz, q_update...)
		shift_vec[1] = q_update[end][1]
	end

	return q_viz
end

q_viz = get_q_viz(q̄)
open(vis)
visualize!(vis, model,
	q_viz,
	Δt = h̄[1])

(q̄[end][1] - q̄[1][1]) / tf_

h̄[1]

model_sim = Walker{Discrete, FixedTime}(n, m, d,
			  g, μ,
			  l_torso, d_torso, m_torso, J_torso,
			  l_thigh, d_thigh, m_thigh, J_thigh,
			  l_calf, d_calf, m_calf, J_calf,
			  l_foot, d_foot, m_foot, J_foot,
			  l_thigh, d_thigh, m_thigh, J_thigh,
			  l_calf, d_calf, m_calf, J_calf,
			  l_foot, d_foot, m_foot, J_foot,
			  qL, qU,
			  uL, uU,
			  nq,
			  nu,
			  nc,
			  nf,
			  nb,
			  ns,
			  idx_u,
			  idx_λ,
			  idx_b,
			  idx_ψ,
			  idx_η,
			  idx_s,
			  joint_friction)

x_sim = [x̄[1]]
q_sim = [x̄[1][1:model.nq], x̄[2][model.nq .+ (1:model.nq)]]
include(joinpath(pwd(), "src/contact_simulator/simulator.jl"))
for t = 1:T-1
	_x = step_contact(model_sim, x_sim[end], ū[t][1:model.nu], zeros(model.d), h̄[t],
	        tol_c = 1.0e-5, tol_opt = 1.0e-5, tol_s = 1.0e-4, nlp = :ipopt)
	push!(x_sim, _x)
	push!(q_sim, x_sim[end][model.nq .+ (1:model.nq)])
end
plot(hcat(q̄...)')
plot(hcat(q_sim...)')
vis = Visualizer()
render(vis)
# open(vis)
visualize!(vis, model, q_sim, Δt = h̄[1])
