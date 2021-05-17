using Plots

# Model
include_model("quadruped_v2")
model = free_time_model(model)

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
open(vis)
# render(vis)

# Horizon
T = 61
Tm = 31
T_fix = 5

# Time step
tf = 1.0
h = tf / (T - 1)

function ellipse_traj(x_start, x_goal, z, T)
	dist = x_goal - x_start
	a = 0.5 * dist
	b = z
	z̄ = 0.0
	# x = range(x_start, stop = x_goal, length = T)
	x = circular_projection_range(x_start, stop = x_goal, length = T)
	z = sqrt.(max.(0.0, (b^2) * (1.0 .- ((x .- (x_start + a)).^2.0) / (a^2.0))))
	return x, z
end

function circular_projection_range(start; stop=1.0, length=10)
	dist = stop - start
	θr = range(π, stop=0, length=length)
	r = start .+ dist * ((1 .+ cos.(θr))./2)
	return r
end


function initial_configuration(model::QuadrupedV2; offset = 0.025)
	q1 = zeros(model.nq)

	# position
	q1[3] = 0.2

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

q_shift1 = zeros(model.nq)
q_shift1[1] = 0.5 * strd
q_shift1[10] = strd
q_shift1[13] = strd

qM = copy(q1) + q_shift1

q_shift2 = zeros(model.nq)
q_shift2[1] = 0.5 * strd
q_shift2[7] = strd
q_shift2[16] = strd

qT = copy(qM) + q_shift2

q_ref = [q1, linear_interpolation(q1, qM, 30)..., linear_interpolation(qM, qT, 30)...]
visualize!(vis, model, q_ref)

zh = 0.05
xf1_el, zf1_el = ellipse_traj(pf1[1], pf1[1] + strd, zh, Tm - T_fix)
xf1 = [[pf1[1] for t = 1:Tm + T_fix]..., xf1_el[2:end]...]
zf1 = [[pf1[3] for t = 1:Tm + T_fix]..., zf1_el[2:end]...]
pf1_ref = [[xf1[t]; pf1[2];  zf1[t]] for t = 1:T]

xf4_el, zf4_el = ellipse_traj(pf4[1], pf4[1] + strd, zh, Tm - T_fix)
xf4 = [[pf4[1] for t = 1:Tm + T_fix]..., xf4_el[2:end]...]
zf4 = [[pf4[3] for t = 1:Tm + T_fix]..., zf4_el[2:end]...]
pf4_ref = [[xf4[t]; pf4[2]; zf4[t]] for t = 1:T]

xf2_el, zf2_el = ellipse_traj(pf2[1], pf2[1] + strd, zh, Tm - T_fix)
xf2 = [[xf2_el[1] for t = 1:T_fix]..., xf2_el..., [xf2_el[end] for t = 1:Tm-1 + T_fix]...]
zf2 = [[zf2_el[1] for t = 1:T_fix]..., zf2_el..., [zf2_el[end] for t = 1:Tm-1 + T_fix]...]
pf2_ref = [[xf2[t]; pf2[2]; zf2[t]] for t = 1:T]

xf3_el, zf3_el = ellipse_traj(pf3[1], pf3[1] + strd, zh, Tm - T_fix)
xf3 = [[xf3_el[1] for t = 1:T_fix]..., xf3_el..., [xf3_el[end] for t = 1:Tm-1]...]
zf3 = [[zf3_el[1] for t = 1:T_fix]..., zf3_el..., [zf3_el[end] for t = 1:Tm-1]...]
pf3_ref = [[xf3[t]; pf3[2]; zf3[t]] for t = 1:T]

tr = range(0, stop = tf, length = T)
plot(tr, hcat(pf1_ref...)')
plot!(tr, hcat(pf4_ref...)')

plot!(tr, hcat(pf2_ref...)')
plot!(tr, hcat(pf3_ref...)')


# Bounds

# control
# ul <= u <= uu
_uu = Inf * ones(model.m)
_uu[model.idx_u] .= Inf
_uu[end] = 1.0 * h
_ul = zeros(model.m)
_ul[model.idx_u] .= -Inf
_ul[end] = 1.0 * h
ul, uu = control_bounds(model, T, _ul, _uu)

xl, xu = state_bounds(model, T,
    x1 = [Inf * ones(model.nq); q1])

# Objective
include_objective(["velocity", "nonlinear_stage", "control_velocity"])
q_ref = linear_interpolation(q1, qT, T+1)
# render(vis)
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
	J += 1000.0 * (q2[3] - 0.2)^2.0

	# orientation
	J += 1000.0 * sum((q2[4:6] - q_ref[t][4:6]).^2.0)

	# forward velocity
	J += 1000.0 * ((q2[1] - q1[1]) / h - strd / tf)^2.0

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
    q = view(x, 18 .+ (1:18))
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
t_idx1 = vcat([t for t = 1:Tm + T_fix])
t_idx2 = vcat([t for t = 1:T_fix]..., [t for t = Tm:T]...)
con_pinned1 = stage_constraints(pinned1!, n_stage, (1:0), t_idx1)
con_pinned2 = stage_constraints(pinned2!, n_stage, (1:0), t_idx2)

con_contact = contact_constraints(model, T)
con_free_time = free_time_constraints(T)
con_loop1 = loop_constraints(model, collect(1:2model.nq), 1, Tm, shift = [q_shift1; q_shift1])
con_loop2 = loop_constraints(model, collect(1:2model.nq), Tm, T, shift = [q_shift2; q_shift2])

con = multiple_constraints([con_contact,
	con_loop1, con_loop2,
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
    tol = 1.0e-2, c_tol = 1.0e-2,
	max_iter = 2000,
    time_limit = 60 * 2, mapl = 5)

check_slack(z̄, prob)
x̄, ū = unpack(z̄, prob)
q̄ = state_to_configuration(x̄)
_tf, _t, h̄ = get_time(ū)
@show h̄[1]


# vis = Visualizer()
# render(vis)
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
plot(hcat(ū...)[1:model.nu, :]', linetype = :steppost, label = "")

# unpack solution
q = state_to_configuration(x̄)
u = [u[model.idx_u] for u in ū]
γ = [u[model.idx_λ] for u in ū]
b = [u[model.idx_b] for u in ū]
ψ = [u[model.idx_ψ] for u in ū]
η = [u[model.idx_η] for u in ū]
hm = mean(h̄)
μm = model.μ

qm = q; um = u; γm = γ; bm = b; ψm = ψ; ηm = η;
@save joinpath(@__DIR__, "quadruped_v2_mirror_gait.jld2") qm um γm bm ψm ηm μm hm

plot(hcat(q...)')
plot(hcat(q...)')

[norm(fd(model, [q[t+1]; q[t+2]], [q[t]; q[t+1]], [u[t]; γ[t]; b[t]; hm], zeros(model.d), hm, t)) for t = 1:T-1]

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
# 	p10 = q[2][6 .+ (1:3)]
# 	p20 = q[2][9 .+ (1:3)]
# 	p30 = q[2][12 .+ (1:3)]
# 	p40 = q[2][15 .+ (1:3)]
#
# 	@show norm(fd(model, [qm[end-1]; qm[end]], [qm[end-2]; qm[end-1]], [um[end]; γm[end]; bm[end]], zeros(model.d), hm, 1))
#
# 	for t = 1:1#T-1
# 		# configuration
# 		pt = q[t+2][1:3]
# 		a = q[t+2][3 .+ (1:3)]
# 		p1 = q[t+2][6 .+ (1:3)]
# 		# p2 = q[t+2][9 .+ (1:3)]
# 		# p3 = q[t+2][12 .+ (1:3)]
# 		p4 = q[t+2][15 .+ (1:3)]
#
# 		p2_diff = q[t+2][9 .+ (1:3)] - p20
# 		p3_diff = q[t+2][12 .+ (1:3)] - p30
#
# 		push!(qm, [pt + [0.5 * strd; 0.0; 0.0]; a;
# 			p10[1] + p2_diff[1]; q[t+2][8]; p10[3] + p2_diff[3]
# 			q[end][9 .+ (1:3)];
# 			q[end][12 .+ (1:3)];
# 			p40[1] + p3_diff[1]; q[t+2][17]; p40[3] + p3_diff[3];
# 			])
# 		# push!(qm, q[t+2])
#
# 		# control
# 		u1 = u[t][1:3]
# 		u2 = u[t][3 .+ (1:3)]
# 		u3 = u[t][6 .+ (1:3)]
# 		u4 = u[t][9 .+ (1:3)]
# 		push!(um, [u2; u1; u4; u3])
#
# 		# impact
# 		γ1 = γ[t][1]
# 		γ2 = γ[t][2]
# 		γ3 = γ[t][3]
# 		γ4 = γ[t][4]
# 		push!(γm, [γ2; γ1; γ4; γ3])
# 		# push!(γm, γ[t])
#
# 		# friction
# 		b1 = b[t][1:4]
# 		b2 = b[t][4 .+ (1:4)]
# 		b3 = b[t][8 .+ (1:4)]
# 		b4 = b[t][12 .+ (1:4)]
# 		push!(bm, [b2; b1; b4; b3])
# 		# push!(bm, b[t])
#
# 		# dual
# 		ψ1 = ψ[t][1]
# 		ψ2 = ψ[t][2]
# 		ψ3 = ψ[t][3]
# 		ψ4 = ψ[t][4]
# 		push!(ψm, [ψ2; ψ1; ψ4; ψ3])
# 		# push!(ψm, ψ[t])
#
# 		# dual
# 		η1 = η[t][1:4]
# 		η2 = η[t][4 .+ (1:4)]
# 		η3 = η[t][8 .+ (1:4)]
# 		η4 = η[t][12 .+ (1:4)]
# 		push!(ηm, [η2; η1; η4; η3])
# 		# push!(ηm, η[t])
#
# 		@show norm(fd(model, [qm[end-1]; qm[end]], [qm[end-2]; qm[end-1]], [um[end]; γm[end]; bm[end]], zeros(model.d), hm, 1))
# 	end
#
# 	return qm, um, γm, bm, ψm, ηm
# end
#
# qm, um, γm, bm, ψm, ηm = mirror_gait(q, u, γ, b, ψ, η, T)
#
# # @save joinpath(@__DIR__, "quadruped_v2_mirror_gait.jld2") qm um γm bm ψm ηm μm hm
#
plot(hcat(q...)', color = :black, width = 2.0, label = "")
plot!(hcat(qm...)', color = :red, width = 1.0, label = "")
#
plot(hcat(u...)', color = :black, width = 2.0, label = "", linetype = :steppost)
# plot!(hcat(um...)', color = :red, width = 1.0, label = "", linetype = :steppost)
#
plot(hcat(γ...)', color = :black, width = 2.0, label = "", linetype = :steppost)
# plot!(hcat(γm...)', color = :red, width = 1.0, label = "", linetype = :steppost)
#
plot(hcat(b...)', color = :black, width = 2.0, label = "", linetype = :steppost)
# plot!(hcat(bm...)', color = :red, width = 1.0, label = "", linetype = :steppost)
# #
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

# vis = Visualizer()
# render(vis)
visualize!(vis, model,
	qm,
	Δt = h̄[1])

settransform!(vis["/Cameras/default"],
        compose(Translation(0.0, -25.0, -1.0), LinearMap(RotY(0.0 * π) * RotZ(-π / 2.0))))
setprop!(vis["/Cameras/default/rotated/<object>"], "zoom", 20)
# open(vis)
