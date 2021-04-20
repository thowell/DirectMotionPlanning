using Plots

# Model
include_model("quadruped")
model = free_time_model(model)

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)

# Horizon
T = 31
T_fix = 5

# Time step
tf = 0.625
h = tf / (T - 1)

# Permutation matrix
perm = @SMatrix [1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
                 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
				 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
				 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0;
				 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0;
				 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
				 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0;
				 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0;
				 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0;
				 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0;
				 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0]

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
qT = Array(perm) * copy(q1)
qT[1] += 0.5 * strd

# torso height
pt = kinematics_1(model, q1, body = :torso, mode = :com)

zh = 0.05
xr1 = [pr1[1] for t = 1:T]
zr1 = [pr1[2] for t = 1:T]
pr1_ref = [[xr1[t]; zr1[t]] for t = 1:T]
xf1 = [pf1[1] for t = 1:T]
zf1 = [pf1[2] for t = 1:T]
pf1_ref = [[xf1[t]; zf1[t]] for t = 1:T]

xr2_el, zr2_el = ellipse_traj(pr2[1], pr2[1] + strd, zh, T - T_fix)
xr2 = [[xr2_el[1] for t = 1:T_fix]..., xr2_el...]
zr2 = [[zr2_el[1] for t = 1:T_fix]..., zr2_el...]
pr2_ref = [[xr2[t]; zr2[t]] for t = 1:T]

xf2_el, zf2_el = ellipse_traj(pf2[1], pf2[1] + strd, zh, T - T_fix)
xf2 = [[xf2_el[1] for t = 1:T_fix]..., xf2_el...]
zf2 = [[zf2_el[1] for t = 1:T_fix]..., zf2_el...]
pf2_ref = [[xf2[t]; zf2[t]] for t = 1:T]

tr = range(0, stop = tf, length = T)
plot(tr, hcat(pr1_ref...)')
plot!(tr, hcat(pf1_ref...)')

plot(tr, hcat(pr2_ref...)')
plot!(tr, hcat(pf2_ref...)')


# Bounds

# control
# ul <= u <= uu
_uu = Inf * ones(model.m)
_uu[model.idx_u] .= 33.5 * h
_uu[end] = 2.0 * h
_ul = zeros(model.m)
_ul[model.idx_u] .= -33.5 * h
_ul[end] = 0.75 * h
ul, uu = control_bounds(model, T, _ul, _uu)

xl, xu = state_bounds(model, T,
    x1 = [Inf * ones(model.nq); q1],
    xT = [Inf * ones(model.nq); qT[1]; Inf * ones(model.nq - 1)])

# Objective
include_objective(["velocity", "nonlinear_stage", "control_velocity"])
q_ref = linear_interpolation(q1, qT, T+1)
render(vis)
visualize!(vis, model, q_ref, Δt = h)
x0 = configuration_to_state(q_ref)

# penalty on slack variable
obj_penalty = PenaltyObjective(1.0e4, model.m - 1)

# quadratic tracking objective
# Σ (x - xref)' Q (x - x_ref) + (u - u_ref)' R (u - u_ref)
obj_control = quadratic_time_tracking_objective(
    [1.0 * Diagonal(1.0e-5 * ones(model.n)) for t = 1:T],
    [1.0 * Diagonal([1.0e-3 * ones(model.nu)..., 1.0e-3 * ones(model.nc + model.nb)..., zeros(model.m - model.nu - model.nc - model.nb)...]) for t = 1:T-1],
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

obj_ctrl_velocity = control_velocity_objective(Diagonal([1.0e-3 * ones(model.nu)..., 1.0e-3 * ones(model.nc + model.nb)..., zeros(model.m - model.nu - model.nc - model.nb)...]))

function l_stage(x, u, t)
	q1 = view(x, 1:11)
	q2 = view(x, 11 .+ (1:11))
    J = 0.0

	# torso height
    J += 100.0 * (kinematics_1(model, q1, body = :torso, mode = :ee)[2] - kinematics_1(model, view(x0[t], 1:11), body = :torso, mode = :com)[2])^2.0
	J += 100.0 * (kinematics_1(model, q2, body = :torso, mode = :ee)[2] - kinematics_1(model, view(x0[t], 1:11), body = :torso, mode = :com)[2])^2.0

	J += 100.0 * (q1[2] - x0[1][2])^2.0
	J += 100.0 * (q2[2] - x0[1][2])^2.0

    if true
		J += 10000.0 * sum((pr2_ref[t] - kinematics_2(model, q1, body = :calf_2, mode = :ee)).^2.0)
	    J += 10000.0 * sum((pf2_ref[t] - kinematics_3(model, q1, body = :calf_4, mode = :ee)).^2.0)
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
    nothing
end

function pinned2!(c, x, u, t)
    q = view(x, 1:11)
	c[1:2] = pr2_ref[t] - kinematics_2(model, q, body = :calf_2, mode = :ee)
    c[3:4] = pf2_ref[t] - kinematics_3(model, q, body = :calf_4, mode = :ee)
    nothing
end

n_stage = 4
t_idx1 = vcat([t for t = 1:T])
t_idx2 = vcat([1:T_fix]...)
con_pinned1 = stage_constraints(pinned1!, n_stage, (1:0), t_idx1)
con_pinned2 = stage_constraints(pinned2!, n_stage, (1:0), t_idx2)

con_contact = contact_constraints(model, T)
con_free_time = free_time_constraints(T)
con_loop = loop_constraints(model, collect([(2:model.nq)...,
	(nq .+ (2:model.nq))...]), 1, T, perm = Array(cat(perm, perm, dims = (1,2))))
con = multiple_constraints([con_contact, con_loop,
    con_free_time, con_pinned1, con_pinned2])

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
u0 = [[1.0e-3 * rand(model.m - 1); h] for t = 1:T-1] # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

# Solve
@time z̄, info = solve(prob, copy(z0),
    nlp = :ipopt,
    tol = 1.0e-3, c_tol = 1.0e-3,
	max_iter = 2000,
    time_limit = 60 * 2, mapl = 5)

check_slack(z̄, prob)
x̄, ū = unpack(z̄, prob)
q̄ = state_to_configuration(x̄)
_tf, _t, h̄ = get_time(ū)
@show h̄[1]


vis = Visualizer()
render(vis)
visualize!(vis, model, state_to_configuration(x̄), Δt = h̄[1])

# check foot trajectories
_pr1_ref = [kinematics_2(model, q, body = :calf_1, mode = :ee) for q in q̄]
_pf1_ref = [kinematics_3(model, q, body = :calf_3, mode = :ee) for q in q̄]

_pr2_ref = [kinematics_2(model, q, body = :calf_2, mode = :ee) for q in q̄]
_pf2_ref = [kinematics_3(model, q, body = :calf_4, mode = :ee) for q in q̄]

plot(hcat(pr1_ref...)', width = 2.0, color = :black)
plot!(hcat(_pr1_ref...)', color = :red)

plot(hcat(pf1_ref...)', width = 2.0, color = :black)
plot!(hcat(_pf1_ref...)', color = :red)

plot(hcat(pr2_ref...)', width = 2.0, color = :black)
plot!(hcat(_pr2_ref...)', color = :red)

plot(hcat(pf2_ref...)', width = 2.0, color = :black)
plot!(hcat(_pf2_ref...)', color = :red)

# check control trajectory
plot(hcat(ū...)[1:model.nu, :]', linetype = :steppost)

# unpack solution
q = state_to_configuration(x̄)
u = [u[model.idx_u] for u in ū]
γ = [u[model.idx_λ] for u in ū]
b = [u[model.idx_b] for u in ū]
ψ = [u[model.idx_ψ] for u in ū]
η = [u[model.idx_η] for u in ū]
hm = mean(h̄)
μm = model.μ

perm4 = perm[end-3:end, end-3:end]
perm8 = perm[end-7:end, end-7:end]

function mirror_gait(q, u, γ, b, ψ, η, T)
	qm = [deepcopy(q)...]
	um = [deepcopy(u)...]
	γm = [deepcopy(γ)...]
	bm = [deepcopy(b)...]
	ψm = [deepcopy(γ)...]
	ηm = [deepcopy(b)...]

	stride = zero(qm[1])
	stride[1] = q[T][1] - q[2][1]

	for t = 1:T-1
		push!(qm, Array(perm) * q[t+2] + stride)
		push!(um, perm8 * u[t])
		push!(γm, perm4 * γ[t])
		push!(bm, perm8 * b[t])
		push!(ψm, perm4 * ψ[t])
		push!(ηm, perm8 * η[t])
	end

	return qm, um, γm, bm, ψm, ηm
end

qm, um, γm, bm, ψm, ηm = mirror_gait(q, u, γ, b, ψ, η, T)

@save joinpath(@__DIR__, "quadruped_mirror_gait.jld2") qm um γm bm ψm ηm μm hm

plot(hcat(q...)', color = :black, width = 2.0, label = "")
plot!(hcat(qm...)', color = :red, width = 1.0, label = "")

plot(hcat(u...)', color = :black, width = 2.0, label = "", linetype = :steppost)
plot!(hcat(um...)', color = :red, width = 1.0, label = "", linetype = :steppost)

plot(hcat(γ...)', color = :black, width = 2.0, label = "", linetype = :steppost)
plot!(hcat(γm...)', color = :red, width = 1.0, label = "", linetype = :steppost)

plot(hcat(b...)', color = :black, width = 2.0, label = "", linetype = :steppost)
plot!(hcat(bm...)', color = :red, width = 1.0, label = "", linetype = :steppost)

function get_q_viz(q̄)
	q_viz = [q̄...]
	shift_vec = zeros(model.nq)
	shift_vec[1] = q̄[end][1]
	for i = 1:5
		q_update = [q + shift_vec for q in q̄[2:end]]
		push!(q_viz, q_update...)
		shift_vec[1] = q_update[end][1]
	end

	return q_viz
end

q_viz = get_q_viz(q̄)
visualize!(vis, model,
	qm,
	Δt = h̄[1])
