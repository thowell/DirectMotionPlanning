using Plots

# Model
include_model("flamingo")

model = free_time_model(model)

# Visualize
# - Pkg.add any external deps from visualize.jl
include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
open(vis)
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

	x = circular_projection_range(x_start, stop = x_goal, length = T)
	# x = range(x_start, stop = x_goal, length = T)

	z = sqrt.(max.(0.0, (b^2) * (1.0 .- ((x .- (x_start + a)).^2.0) / (a^2.0))))

	return x, z
end

function circular_projection_range(start; stop=1.0, length=10)
	dist = stop - start
	θr = range(π, stop=0, length=length)
	r = start .+ dist * ((1 .+ cos.(θr))./2)
	return r
end


# Horizon
# T = 31
T = 36 #CHANGED
# T_fix = 7
T_fix = 10 #CHANGED

# Time step
tf = 0.5
# tf = 0.9 #CHANGED
h = tf / (T - 1)

# Permutation matrix
perm = @SMatrix [1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
                 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
				 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0;
				 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0;
				 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0;
				 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0;
				 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0;
				 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0;
				 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0]

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
    q1[7] = 1.0 * acos((z1 - model.l_thigh2 * cos(q1[6])) / model.l_calf2)
	q1[2] = z1
	q1[8] = pi / 2.0
	q1[9] = pi / 2.0

    return q1
end

# q1 = initial_configuration(model, -π / 50.0, -π / 6.5, π / 10.0 , -π / 10.0)
# q1 = initial_configuration(model, -0.00*π / 50.0, -π / 5.6, π / 25.0 , -π / 7.0)
q1 = initial_configuration(model, -0.00*π / 50.0, -π / 5.6, π / 15.0 , -π / 7.0)
pf1 = kinematics_3(model, q1, body = :foot_1, mode = :com)
pf2 = kinematics_3(model, q1, body = :foot_2, mode = :com)

ph1 = kinematics_3(model, q1, body = :foot_1, mode = :heel)
ph2 = kinematics_3(model, q1, body = :foot_2, mode = :heel)[1]
ph2 = kinematics_2(model, q1, body = :calf_2, mode = :ee)[1]
q1[1]

strd = 2 * (pf2 - pf1)[1]

qT = Array(perm) * copy(q1)
qT[1] += 0.5 * strd
q_ref = linear_interpolation(q1, qT, T+1)
x_ref = configuration_to_state(q_ref)
visualize!(vis, model, q_ref, Δt = h)

zh = 0.1
xf1_el, zf1_el = ellipse_traj(pf1[1], pf1[1] + strd, zh, T - T_fix)
xf2_el, zf2_el = ellipse_traj(pf2[1], pf2[1] + strd, zh, T - T_fix)

zf1 = [[zf1_el[1] for t = 1:T_fix]..., zf1_el...]
xf1 = [[xf1_el[1] for t = 1:T_fix]..., xf1_el...]
zf2 = [zf2_el[1] for t = 1:T]
xf2 = [xf2_el[1] for t = 1:T]

p1_ref = [[xf1[t]; zf1[t]] for t = 1:T]
p2_ref = [[xf2[t]; zf2[t]] for t = 1:T]

plot(hcat(p1_ref...)'[:,1:1], legend = :topleft)
plot(hcat(p1_ref...)'[:,2:2], legend = :topleft)
plot(hcat(p1_ref...)', legend = :topleft)
plot!(hcat(p2_ref...)')

# Control

# ul <= u <= uu
_uu = Inf * ones(model.m)
_uu[model.idx_u] .= Inf
_uu[end] = 2.0 * h
_ul = zeros(model.m)
_ul[model.idx_u] .= -Inf
_ul[end] = 0.25 * h
ul, uu = control_bounds(model, T, _ul, _uu)

qL = [-Inf; -Inf; q1[3] - pi / 750.0; q1[4:end] .- pi / 6.0; -Inf; -Inf; q1[3] - pi / 50.0; q1[4:end] .- pi / 6.0]
qU = [Inf; q1[2] + 0.001; q1[3] + pi / 750.0; q1[4:end] .+ pi / 6.0; Inf; Inf; q1[3:end] .+ pi / 6.0]
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
obj_penalty = PenaltyObjective(1.0e5, model.m - 1)

# quadratic tracking objective
# Σ (x - xref)' Q (x - x_ref) + (u - u_ref)' R (u - u_ref)
q_penalty = 1.0e-2 * ones(model.nq)

x_penalty = [q_penalty; q_penalty]
obj_control = quadratic_time_tracking_objective(
    [Diagonal(x_penalty) for t = 1:T],
    [Diagonal([1.0e-2 * ones(model.nu)..., 1.0e-3 * ones(model.nc)..., 1.0e-3 * ones(model.nb)..., 1.0e-8 * ones(model.m - model.nu - model.nc - model.nb - 1)..., 0.0]) for t = 1:T-1],
    [x_ref[end] for t = 1:T],
    [[zeros(model.nu); zeros(model.m - model.nu)] for t = 1:T-1],
	1.0)

obj_ctrl_vel = control_velocity_objective(Diagonal([1.0e-1 * ones(model.nu); 1.0 * ones(model.nc + model.nb); zeros(model.m - model.nu - model.nc - model.nb)]))

# quadratic velocity penalty
q_v = 1.0e-1 * ones(model.nq)
obj_velocity = velocity_objective(
    [Diagonal(q_v) for t = 1:T-1],
    model.nq,
    h = h)

function l_foot_height(x, u, t)
	J = 0.0
	q1 = view(x, 1:9)

	if true
		J += 10000.0 * sum((p1_ref[t] - kinematics_3(model, q1, body = :foot_1, mode = :toe)).^2.0)
		J += 10000.0 * sum((p1_ref[t] - kinematics_3(model, q1, body = :foot_1, mode = :heel)).^2.0)
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
    c[1:2] = p1_ref[t] - kinematics_3(model, q1, body = :foot_1, mode = :com)
	nothing
end

function pinned2!(c, x, u, t)
    q1 = view(x, 1:9)
	c[1:2] = p2_ref[t] - kinematics_3(model, q1, body = :foot_2, mode = :com)
	nothing
end

n_stage = 2
t_idx1 = vcat([t for t = 1:T_fix]..., T)
t_idx2 = vcat([1:T]...)

con_pinned1 = stage_constraints(pinned1!, n_stage, (1:0), t_idx1)
con_pinned2 = stage_constraints(pinned2!, n_stage, (1:0), t_idx2)

con_loop = loop_constraints(model, collect([(2:9)...,(11:18)...]), 1, T, perm = Array(cat(perm, perm, dims = (1:2))))
con_contact = contact_constraints(model, T)
con_free_time = free_time_constraints(T)
con = multiple_constraints([con_contact,
	con_free_time,
	con_loop,
	con_pinned1,
	con_pinned2])

# Problem
prob = trajectory_optimization_problem(model,
               obj,
               T,
               xl = xl,
               xu = xu,
               ul = ul,
               uu = uu,
               con = con)

# trajectory initialization
u0 = [[1.0e-2 * randn(model.nu); 0.01 * randn(model.m - model.nu - 1); h] for t = 1:T-1] # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

# Solve
# NOTE: run multiple times to get good trajectory
@time z̄, info = solve(prob, copy(z0),
    nlp = :ipopt,
	max_iter = 2000,
    tol = 1.0e-3, c_tol = 1.0e-3, mapl = 5,
    time_limit = 60 * 3)

@show check_slack(z̄, prob)
x̄, ū = unpack(z̄, prob)
_tf, _t, h̄ = get_time(ū)

q = state_to_configuration(x̄)
u = [u[model.idx_u] for u in ū]
γ = [u[model.idx_λ] for u in ū]
b = [u[model.idx_b] for u in ū]
ψ = [u[model.idx_ψ] for u in ū]
η = [u[model.idx_η] for u in ū]
h̄ = mean(h̄)

plot(hcat(u...)', linetype = :steppost)

_pf1 = [kinematics_3(model, qt, body = :foot_1, mode = :com) for qt in q]
_pf2 = [kinematics_3(model, qt, body = :foot_2, mode = :com) for qt in q]

plot(hcat(p1_ref...)', title = "foot 1",
 	legend = :topleft, label = ["x" "z"], color = :black, width = 2.0)
plot!(hcat(_pf1...)', legend = :topleft, color = :red, width = 1.0)

plot(hcat(p2_ref...)', title = "foot 2",
	legend = :topleft, label = ["x" "z"], color = :black, width = 2.0)
plot!(hcat(_pf2...)', title = "foot 2", legend = :bottomright,
	color = :red, width = 1.0)

perm4 = [0.0 0.0 1.0 0.0;
         0.0 0.0 0.0 1.0;
		 1.0 0.0 0.0 0.0;
		 0.0 1.0 0.0 0.0]

perm7 = [0.0 0.0 1.0 0.0 0.0 0.0;
		 0.0 0.0 0.0 1.0 0.0 0.0;
		 1.0 0.0 0.0 0.0 0.0 0.0;
		 0.0 1.0 0.0 0.0 0.0 0.0;
		 0.0 0.0 0.0 0.0 0.0 1.0;
		 0.0 0.0 0.0 0.0 1.0 0.0]

perm8 = [0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0;
         0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0;
		 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0;
		 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0;
		 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
		 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0;
		 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0;
		 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0]

function mirror_gait(q, u, γ, b, ψ, η, T)
	qm = [deepcopy(q)...]
	um = [deepcopy(u)...]
	γm = [deepcopy(γ)...]
	bm = [deepcopy(b)...]
	ψm = [deepcopy(ψ)...]
	ηm = [deepcopy(η)...]

	stride = zero(qm[1])
	@show stride[1] = q[T+1][1] - q[2][1]
	@show 0.5 * strd
	for t = 1:T-1
		push!(qm, Array(perm) * q[t+2] + stride)
		push!(um, perm7 * u[t])
		push!(γm, perm4 * γ[t])
		push!(bm, perm8 * b[t])
		push!(ψm, perm4 * ψ[t])
		push!(ηm, perm8 * η[t])
	end

	return qm, um, γm, bm, ψm, ηm
end

hm = h̄
μm = model.μ
qm, um, γm, bm, ψm, ηm = mirror_gait(q, u, γ, b, ψ, η, T)

@save joinpath(@__DIR__, "flamingo_mirror_gait.jld2") qm um γm bm ψm ηm μm hm

vis = Visualizer()
render(vis)
open(vis)
visualize!(vis, model,
	qm,
	Δt = h̄[1])
