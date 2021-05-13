using Plots

# Model
include_model("quadruped_v2")
model = free_time_model(model)

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)

# Horizon
T = 31
T_fix = 5

# Time step
tf = 1.5
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

function initial_configuration(model::QuadrupedV2; offset = 0.1)
	q1 = zeros(model.nq)

	# position
	q1[3] = 0.5

	# orientation
	mrp = MRP(RotZ(0.0))
	q1[4:6] = [mrp.x; mrp.y; mrp.z]

	# feet positions (in body frame)
	q1[7:9] = [model.l_torso + offset; model.w_torso; 0.0]
	q1[10:12] = [model.l_torso - offset; -model.w_torso; 0.0]
	q1[13:15] = [-model.l_torso - offset; model.w_torso; 0.0]
	q1[16:18] = [-model.l_torso + offset; -model.w_torso; 0.0]

	return q1
end

q1 = initial_configuration(model)
visualize!(vis, model, [q1])

# feet positions
pf1 = q1[6 .+ (1:3)]
pf2 = q1[9 .+ (1:3)]

pf3 = q1[12 .+ (1:3)]
pf4 = q1[15 .+ (1:3)]


strd = 2 * (pf1 - pf2)[1]

q_shift = zeros(model.nq)
q_shift[1] = 0.5 * strd
q_shift[10] = strd
q_shift[13] = strd

qT = copy(q1) + q_shift

visualize!(vis, model, linear_interpolation(q1, qT, 10))

zh = 0.1
xf1 = [pf1[1] for t = 1:T]
zf1 = [pf1[3] for t = 1:T]
pf1_ref = [[xf1[t]; pf1[2];  zf1[t]] for t = 1:T]
xf4 = [pf4[1] for t = 1:T]
zf4 = [pf4[3] for t = 1:T]
pf4_ref = [[xf4[t]; pf4[2]; zf4[t]] for t = 1:T]

xf2_el, zf2_el = ellipse_traj(pf2[1], pf2[1] + strd, zh, T - T_fix)
xf2 = [[xf2_el[1] for t = 1:T_fix]..., xf2_el...]
zf2 = [[zf2_el[1] for t = 1:T_fix]..., zf2_el...]
pf2_ref = [[xf2[t]; pf2[2]; zf2[t]] for t = 1:T]

xf3_el, zf3_el = ellipse_traj(pf3[1], pf3[1] + strd, zh, T - T_fix)
xf3 = [[xf3_el[1] for t = 1:T_fix]..., xf3_el...]
zf3 = [[zf3_el[1] for t = 1:T_fix]..., zf3_el...]
pf3_ref = [[xf3[t]; pf3[2]; zf3[t]] for t = 1:T]

tr = range(0, stop = tf, length = T)
plot(tr, hcat(pf1_ref...)')
plot!(tr, hcat(pf4_ref...)')

plot(tr, hcat(pf2_ref...)')
plot!(tr, hcat(pf3_ref...)')


# Bounds

# control
# ul <= u <= uu
_uu = Inf * ones(model.m)
_uu[model.idx_u] .= Inf
_uu[end] = h
_ul = zeros(model.m)
_ul[model.idx_u] .= -Inf
_ul[end] = h
ul, uu = control_bounds(model, T, _ul, _uu)

xl, xu = state_bounds(model, T,
    x1 = [Inf * ones(model.nq); q1])

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
    [1.0 * Diagonal(1.0e-1 * ones(model.n)) for t = 1:T],
    [1.0 * Diagonal([1.0e-1 * ones(model.nu)..., 1.0e-3 * ones(model.nc + model.nb)..., zeros(model.m - model.nu - model.nc - model.nb)...]) for t = 1:T-1],
    [[qT; qT] for t = 1:T],
    [zeros(model.m) for t = 1:T],
    1.0)

# quadratic velocity penalty
#Σ v' Q v
v_penalty = 1.0e-2 * ones(model.nq)
obj_velocity = velocity_objective(
    [h * Diagonal(v_penalty) for t = 1:T-1],
    model.nq,
    h = h,
    idx_angle = collect([3, 4, 5]))

obj_ctrl_velocity = control_velocity_objective(Diagonal([1.0e-3 * ones(model.nu)..., 1.0e-3 * ones(model.nc + model.nb)..., zeros(model.m - model.nu - model.nc - model.nb)...]))

function l_stage(x, u, t)
	q1 = view(x, 1:18)
	q2 = view(x, 18 .+ (1:18))
    J = 0.0

	# torso height
	J += 100.0 * (q2[3] - 0.5)^2.0

    if true
		p1 = q2[6 .+ (1:3)]
		p2 = q2[9 .+ (1:3)]
		p3 = q2[12 .+ (1:3)]
		p4 = q2[15 .+ (1:3)]

		J += 1000.0 * sum((pf1_ref[t] - p1).^2.0)
		J += 1000.0 * sum((pf2_ref[t] - p2).^2.0)
		J += 1000.0 * sum((pf3_ref[t] - p3).^2.0)
		J += 1000.0 * sum((pf4_ref[t] - p4).^2.0)
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
    q = view(x, 1:18)
    c[1:3] = pf1_ref[t] - q[6 .+ (1:3)]
    c[4:6] = pf4_ref[t] - q[15 .+ (1:3)]
    nothing
end

function pinned2!(c, x, u, t)
    q = view(x, 1:18)
	c[1:3] = pf2_ref[t] - q[9 .+ (1:3)]
    c[4:6] = pf3_ref[t] - q[12 .+ (1:3)]
    nothing
end

n_stage = 6
t_idx1 = vcat([t for t = 1:T])
t_idx2 = vcat([1:T_fix]...)
con_pinned1 = stage_constraints(pinned1!, n_stage, (1:0), t_idx1)
con_pinned2 = stage_constraints(pinned2!, n_stage, (1:0), t_idx2)

con_contact = contact_constraints(model, T)
con_free_time = free_time_constraints(T)
con_loop = loop_constraints(model, collect(1:2model.nq), 1, T, shift = [q_shift; q_shift])

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
_pf1_ref = [q[6 .+ (1:3)] for q in q̄]
_pf2_ref = [q[9 .+ (1:3)]  for q in q̄]

_pf3_ref = [q[12 .+ (1:3)]  for q in q̄]
_pf4_ref = [q[15 .+ (1:3)]  for q in q̄]

plot(hcat(pf1_ref...)', width = 2.0, color = :black)
plot!(hcat(_pf1_ref...)', color = :red)

plot(hcat(pf2_ref...)', width = 2.0, color = :black)
plot!(hcat(_pf2_ref...)', color = :red)

plot(hcat(pf3_ref...)', width = 2.0, color = :black)
plot!(hcat(_pf3_ref...)', color = :red)

plot(hcat(pf4_ref...)', width = 2.0, color = :black)
plot!(hcat(_pf4_ref...)', color = :red)

# check control trajectory
plot(hcat(ū...)[1:model.nu, :]', linetype = :steppost)

qT[15 .+ (1:3)]
q̄[end][15 .+ (1:3)]

# unpack solution
q = state_to_configuration(x̄)
u = [u[model.idx_u] for u in ū]
γ = [u[model.idx_λ] for u in ū]
b = [u[model.idx_b] for u in ū]
ψ = [u[model.idx_ψ] for u in ū]
η = [u[model.idx_η] for u in ū]
hm = mean(h̄)
μm = model.μ
#
# perm4 = [0.0 1.0 0.0 0.0;
#          1.0 0.0 0.0 0.0;
# 		 0.0 0.0 0.0 1.0;
# 		 0.0 0.0 1.0 0.0]
#
# perm8 = [0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0;
#          0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0;
# 		 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
# 		 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0;
# 		 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0;
# 		 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0;
# 		 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0;
# 		 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0]
#
# function mirror_gait(q, u, γ, b, ψ, η, T)
# 	qm = [deepcopy(q)...]
# 	um = [deepcopy(u)...]
# 	γm = [deepcopy(γ)...]
# 	bm = [deepcopy(b)...]
# 	ψm = [deepcopy(ψ)...]
# 	ηm = [deepcopy(η)...]
#
# 	stride = zero(qm[1])
# 	@show stride[1] = q[T+1][1] - q[2][1]
# 	@show 0.5 * strd
#
# 	for t = 1:T-1
# 		push!(qm, Array(perm) * q[t+2] + stride)
# 		push!(um, perm8 * u[t])
# 		push!(γm, perm4 * γ[t])
# 		push!(bm, perm8 * b[t])
# 		push!(ψm, perm4 * ψ[t])
# 		push!(ηm, perm8 * η[t])
# 	end
#
# 	return qm, um, γm, bm, ψm, ηm
# end
#
# qm, um, γm, bm, ψm, ηm = mirror_gait(q, u, γ, b, ψ, η, T)
#
# @save joinpath(@__DIR__, "quadruped_mirror_gait.jld2") qm um γm bm ψm ηm μm hm
#
# plot(hcat(q...)', color = :black, width = 2.0, label = "")
# plot!(hcat(qm...)', color = :red, width = 1.0, label = "")
#
# plot(hcat(u...)', color = :black, width = 2.0, label = "", linetype = :steppost)
# plot!(hcat(um...)', color = :red, width = 1.0, label = "", linetype = :steppost)
#
# plot(hcat(γ...)', color = :black, width = 2.0, label = "", linetype = :steppost)
# plot!(hcat(γm...)', color = :red, width = 1.0, label = "", linetype = :steppost)
#
# plot(hcat(b...)', color = :black, width = 2.0, label = "", linetype = :steppost)
# plot!(hcat(bm...)', color = :red, width = 1.0, label = "", linetype = :steppost)
#
# function get_q_viz(q̄)
# 	q_viz = [q̄...]
# 	shift_vec = zeros(model.nq)
# 	shift_vec[1] = q̄[end][1]
# 	for i = 1:5
# 		q_update = [q + shift_vec for q in q̄[2:end]]
# 		push!(q_viz, q_update...)
# 		shift_vec[1] = q_update[end][1]
# 	end
#
# 	return q_viz
# end
#
# vis = Visualizer()
# render(vis)
# visualize!(vis, model,
# 	qm,
# 	Δt = h̄[1])
