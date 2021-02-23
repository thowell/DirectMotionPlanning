# Model
include_model("quadruped")
model = free_time_model(model)

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)

# Horizon
T = 60
Tm = 31

# Time step
tf = 0.5
h = tf / (T - 1)

function ellipse_traj(x_start, x_goal, z, T)
	dist = x_goal - x_start
	a = 0.5 * dist
	b = z
	z̄ = 0.0
	x = range(x_start, stop = x_goal, length = T)
	z = sqrt.(max.(0.0, (b^2) * (1.0 .- ((x .- (x_start + a)).^2.0) / (a^2.0))))
	return x, z
end

function initial_configuration(model::Quadruped, θ1, θ2, θ3)
    q1 = zeros(model.nq)
    q1[3] = pi / 2.0
    q1[4] = -θ1
    q1[5] = θ2

    q1[8] = -θ1
    q1[9] = θ2

    q1[2] = model.l_thigh1 * cos(q1[4]) + model.l_calf1 * cos(q1[5])

    q1[10] = -θ3
    q1[11] = acos((q1[2] - model.l_thigh2 * cos(q1[10])) / model.l_calf2)

    q1[6] = -θ3
    q1[7] = acos((q1[2] - model.l_thigh2 * cos(q1[6])) / model.l_calf2)

    return q1
end

θ1 = pi / 4.0
θ2 = pi / 4.0
θ3 = pi / 3.0

q1 = initial_configuration(model, θ1, θ2, θ3)
visualize!(vis, model, [q1])

# feet positions
pr1 = kinematics_2(model, q1, body = :calf_1, mode = :ee)
pr2 = kinematics_2(model, q1, body = :calf_2, mode = :ee)

pf1 = kinematics_3(model, q1, body = :calf_3, mode = :ee)
pf2 = kinematics_3(model, q1, body = :calf_4, mode = :ee)

strd = 2 * (pr1 - pr2)[1]
# strd = 2 * (pf1 - pf2)[1]
qT = copy(q1)
qT[1] += strd

# torso height
pt = kinematics_1(model, q1, body = :torso, mode = :com)

zh = 0.05
xr1_el, zr1_el = ellipse_traj(pr1[1], pr1[1] + strd, zh, Tm)
xr1 = [[xr1_el[1] for t = 1:Tm-1]..., xr1_el...]
zr1 = [[zr1_el[1] for t = 1:Tm-1]..., zr1_el...]
pr1_ref = [[xr1[t]; zr1[t]] for t = 1:T]

xf1_el, zf1_el = ellipse_traj(pf1[1], pf1[1] + strd, zh, Tm)
xf1 = [[xf1_el[1] for t = 1:Tm-1]..., xf1_el...]
zf1 = [[zf1_el[1] for t = 1:Tm-1]..., zf1_el...]
pf1_ref = [[xf1[t]; zf1[t]] for t = 1:T]


xr2_el, zr2_el = ellipse_traj(pr2[1], pr2[1] + strd, zh, Tm)
xr2 = [xr2_el..., [xr2_el[end] for t = 1:Tm-1]...]
zr2 = [zr2_el..., [zr2_el[end] for t = 1:Tm-1]...]
pr2_ref = [[xr2[t]; zr2[t]] for t = 1:T]

xf2_el, zf2_el = ellipse_traj(pf2[1], pf2[1] + strd, zh, Tm)
xf2 = [xf2_el..., [xf2_el[end] for t = 1:Tm-1]...]
zf2 = [zf2_el..., [zf2_el[end] for t = 1:Tm-1]...]
pf2_ref = [[xf2[t]; zf2[t]] for t = 1:T]

using Plots
plot(hcat(pr1_ref...)')
plot!(hcat(pf1_ref...)')

plot(hcat(pr2_ref...)')
plot!(hcat(pf2_ref...)')


# Bounds

# control
# u1 = initial_torque(model, q1, h)[model.idx_u]
# ul <= u <= uu
_uu = Inf * ones(model.m)
_uu[model.idx_u] .= 33.5
_uu[end] = 2.0 * h
_ul = zeros(model.m)
_ul[model.idx_u] .= -33.5
_ul[end] = 0.25 * h
ul, uu = control_bounds(model, T, _ul, _uu)

xl, xu = state_bounds(model, T,
    x1 = [Inf * ones(model.nq); q1],
    xT = [Inf * ones(model.nq); qT[1]; Inf * ones(model.nq - 1)])

# Objective
include_objective(["velocity", "nonlinear_stage", "control_velocity"])
q_ref = linear_interpolation(q1, qT, T)
render(vis)
visualize!(vis, model, q_ref, Δt = h)
x0 = configuration_to_state(q_ref)

# penalty on slack variable
obj_penalty = PenaltyObjective(1.0e4, model.m - 1)

# quadratic tracking objective
# Σ (x - xref)' Q (x - x_ref) + (u - u_ref)' R (u - u_ref)
obj_control = quadratic_time_tracking_objective(
    [1.0 * Diagonal(1.0e-5 * ones(model.n)) for t = 1:T],
    [1.0 * Diagonal([1.0e-5 * ones(model.nu)..., zeros(model.m - model.nu)...]) for t = 1:T-1],
    [[qT; qT] for t = 1:T],
    [zeros(model.m) for t = 1:T],
    1.0)

# quadratic velocity penalty
#Σ v' Q v
v_penalty = 0.0 * ones(model.nq)
obj_velocity = velocity_objective(
    [h * Diagonal(v_penalty) for t = 1:T-1],
    model.nq,
    h = h,
    idx_angle = collect([3, 4, 5, 6, 7, 8, 9, 10, 11]))

obj_ctrl_velocity = control_velocity_objective(Diagonal([1.0e-1 * ones(model.nu); 0.0 * ones(model.m - model.nu)]))

function l_stage(x, u, t)
	q1 = view(x, 1:11)
	q2 = view(x, 11 .+ (1:11))
    J = 0.0

	# torso height
    J += 10.0 * (kinematics_1(model, q1, body = :torso, mode = :ee)[2] - kinematics_1(model, view(x0[t], 1:11), body = :torso, mode = :com)[2])^2.0
	J += 10.0 * (kinematics_1(model, q2, body = :torso, mode = :ee)[2] - kinematics_1(model, view(x0[t], 1:11), body = :torso, mode = :com)[2])^2.0

	# feet height
    if t >= Tm
		# pr1 = kinematics_2(model, q1, body = :calf_1, mode = :ee)
		# pr2 = kinematics_2(model, q2, body = :calf_1, mode = :ee)
		# vr = (pr2 - pr1) ./ h
		#
		# pf1 = kinematics_3(model, q1, body = :calf_3, mode = :ee)
		# pf2 = kinematics_3(model, q2, body = :calf_3, mode = :ee)
		# vf = (pf2 - pf1) ./ h

		J += 100.0 * sum((pr1_ref[t] - kinematics_2(model, q1, body = :calf_1, mode = :ee)).^2.0)
	    J += 100.0 * sum((pf1_ref[t] - kinematics_3(model, q1, body = :calf_3, mode = :ee)).^2.0)
		J += 100.0 * sum((pr1_ref[t] - kinematics_2(model, q2, body = :calf_1, mode = :ee)).^2.0)
	    J += 100.0 * sum((pf1_ref[t] - kinematics_3(model, q2, body = :calf_3, mode = :ee)).^2.0)
		# J += 1.0 * vr' * vr
		# J += 1.0 * vf' * vf
	end

    if t <= Tm
		# pr1 = kinematics_2(model, q1, body = :calf_2, mode = :ee)
		# pr2 = kinematics_2(model, q2, body = :calf_2, mode = :ee)
		# vr = (pr2 - pr1) ./ h
		#
		# pf1 = kinematics_3(model, q1, body = :calf_4, mode = :ee)
		# pf2 = kinematics_3(model, q2, body = :calf_4, mode = :ee)
		# vf = (pf2 - pf1) ./ h

		J += 100.0 * sum((pr2_ref[t] - kinematics_2(model, q1, body = :calf_2, mode = :ee)).^2.0)
	    J += 100.0 * sum((pf2_ref[t] - kinematics_3(model, q1, body = :calf_4, mode = :ee)).^2.0)
		J += 100.0 * sum((pr2_ref[t] - kinematics_2(model, q2, body = :calf_2, mode = :ee)).^2.0)
	    J += 100.0 * sum((pf2_ref[t] - kinematics_3(model, q2, body = :calf_4, mode = :ee)).^2.0)
		# J += 1.0 * vr' * vr
		# J += 1.0 * vf' * vf
	end

    return J
end

l_terminal(x) = 0.0
obj_shaping = nonlinear_stage_objective(l_stage, l_terminal)

obj = MultiObjective([obj_penalty,
                      obj_control,
                      obj_velocity,
					  obj_shaping,
					  obj_ctrl_velocity])
# Constraints
include_constraints(["stage", "contact", "free_time", "loop"])
function pinned1!(c, x, u, t)
    q = view(x, 1:11)
    c[1:2] = pr1_ref[t] - kinematics_2(model, q, body = :calf_1, mode = :ee)
    c[3:4] = pf1_ref[t] - kinematics_3(model, q, body = :calf_3, mode = :ee)
	# c[5:6] = pr2_ref[t] - kinematics_2(model, q, body = :calf_2, mode = :ee)
    # c[7:8] = pf2_ref[t] - kinematics_3(model, q, body = :calf_4, mode = :ee)
    nothing
end

function pinned2!(c, x, u, t)
    q = view(x, 1:11)
    # c[1:2] = pr1_ref[t] - kinematics_2(model, q, body = :calf_1, mode = :ee)
    # c[3:4] = pf1_ref[t] - kinematics_3(model, q, body = :calf_3, mode = :ee)
	c[1:2] = pr2_ref[t] - kinematics_2(model, q, body = :calf_2, mode = :ee)
    c[3:4] = pf2_ref[t] - kinematics_3(model, q, body = :calf_4, mode = :ee)
    nothing
end
#
# pt
n_stage = 4
t_idx1 = [t for t = 1:Tm]
t_idx2 = [t for t = Tm:T-1]
con_pinned1 = stage_constraints(pinned1!, n_stage, (1:0), t_idx1)
con_pinned2 = stage_constraints(pinned2!, n_stage, (1:0), t_idx2)

con_contact = contact_constraints(model, T)
con_free_time = free_time_constraints(T)
con_loop = loop_constraints(model, collect([(2:model.nq)..., (nq .+ (2:model.nq))...]), 1, T)
con = multiple_constraints([con_contact,
    con_free_time, con_loop,
	con_pinned1, con_pinned2])

# cc = zeros(4)
# xx = rand(model.n)
# uu = rand(model.m)

# pinned2!(cc, xx, uu)

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
u0 = [[1.0e-2 * rand(model.m - 1); h] for t = 1:T-1] # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob)
# z0 .+= 0.001 * randn(prob.num_var)

# Solve
include_snopt()
# @load joinpath(@__DIR__, "quadruped_gait.jld2") z̄ q̄ ū τ̄ λ̄ b̄ h̄

@time z̄, info = solve(prob, copy(z0),
    nlp = :ipopt,
    tol = 1.0e-3, c_tol = 1.0e-3,
	max_iter = 1000,
    time_limit = 60 * 2, mapl = 5)

check_slack(z̄, prob)
x̄, ū = unpack(z̄, prob)
q̄ = state_to_configuration(x̄)
τ̄ = [u[model.idx_u] for u in ū]
λ̄ = [u[model.idx_λ] for u in ū]
b̄ = [u[model.idx_b] for u in ū]

_tf, _t, h̄ = get_time(ū)
#
# maximum([norm(fd(model, x̄[t+1], x̄[t], ū[t], zeros(model.d), h̄[t], t)) for t = 1:T-1])
# _ϕ = [minimum(min.(0.0, ϕ_func(model, q))) for q in q̄]
#
# cc = zeros(con_contact.n)
# constraints!(cc, z̄, con_contact, model, prob.prob.idx, h̄[1], T)
# idx_ineq = con_ineq_contact(model, T)
# idx_eq = convert.(Int, setdiff(range(1, stop = con_contact.n, length = con_contact.n), idx_ineq))
# norm(cc[idx_eq], Inf)
# norm(min.(0.0, cc[idx_ineq]), Inf)
#
# length(idx_ineq)
# con_contact.n
# setdiff([(1:5)...], [(4:7)...])
# Visualize
vis = Visualizer()
render(vis)
# open(vis)
visualize!(vis, model, state_to_configuration(x̄), Δt = h̄[1])

_pr1_ref = [kinematics_2(model, q, body = :calf_1, mode = :ee) for q in q̄]
_pf1_ref = [kinematics_3(model, q, body = :calf_3, mode = :ee) for q in q̄]

_pr2_ref = [kinematics_2(model, q, body = :calf_2, mode = :ee) for q in q̄]
_pf2_ref = [kinematics_3(model, q, body = :calf_4, mode = :ee) for q in q̄]

using Plots
plot(hcat(_pr1_ref...)')
plot(hcat(_pf1_ref...)')

plot(hcat(_pr2_ref...)')
plot(hcat(_pf2_ref...)')

plot(hcat(ū...)[1:model.nu, :]', linetype = :steppost)

@save joinpath(@__DIR__, "quadruped_gait.jld2") z̄ x̄ ū q̄ τ̄ λ̄ b̄ h̄
h̄[1]
function get_q_viz(q̄)
	q_viz = [q̄...]
	shift_vec = zeros(model.nq)
	shift_vec[1] = q̄[end][1]
	for i = 1:5
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
visualize!(vis, model,
	q_viz,
	Δt = h̄[1])


# resimulate
model_sim = Quadruped{Discrete, FixedTime}(n, m, d,
			  g, μ,
			  l_torso, d_torso, m_torso, J_torso,
			  l_thigh, d_thigh, m_thigh, J_thigh,
			  l_leg, d_leg, m_leg, J_leg,
			  l_thigh, d_thigh, m_thigh, J_thigh,
			  l_leg, d_leg, m_leg, J_leg,
			  l_thigh, d_thigh, m_thigh, J_thigh,
			  l_leg, d_leg, m_leg, J_leg,
			  l_thigh, d_thigh, m_thigh, J_thigh,
			  l_leg, d_leg, m_leg, J_leg,
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
	_x = step_contact(model_sim, x_sim[end], ū[t][1:model.nu], zeros(model.d), h̄[t],
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
