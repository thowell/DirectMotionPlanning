# Model
include(joinpath(pwd(), "models/quadruped.jl"))
model = Quadruped{Discrete, FixedTime}(n, m, d,
			  g, 0.1,
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
model_ft = free_time_model(model)

# Horizon
T = 41

# Time step
tf = 0.5
h = tf / (T - 1)

# Visualize
# - Pkg.add any external deps from visualize.jl
include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)

# Configurations
θ = pi / 2.5
q1 = initial_configuration(model_ft,  θ)
q1_high = initial_configuration(model_ft,  π / 3.5)

qM = copy(q1)
qM[1] -= 0.5 * model_ft.l_torso
qM[2] += 0.75
# qM[3] += pi
# qM[4] += pi
# qM[5] += pi
# qM[6] += pi
# qM[7] += pi
# qM[8] += pi
# qM[9] += pi
# qM[10] += pi
# qM[11] += pi
qT = copy(q1_high)
qT[1] -= 2 * model_ft.l_torso
# qT[3] += 2.0 * pi
# qT[4] += 2.0 * pi
# qT[5] += 2.0 * pi
# qT[6] += 2.0 * pi
# qT[7] += 2.0 * pi
# qT[8] += 2.0 * pi
# qT[9] += 2.0 * pi
# qT[10] += 2.0 * pi
# qT[11] += 2.0 * pi

visualize!(vis, model_ft, [q1])
visualize!(vis, model_ft, [qM])
visualize!(vis, model_ft, [qT])

q_ref = [linear_interpolation(q1, qM, 21)...,
    linear_interpolation(qM, qT, 21)...]

visualize!(vis, model_ft, q_ref)

# Bounds

# control
# u = (τ1..4, λ1..2, β1..4, ψ1..2, η1...4, s1)
# ul <= u <= uu
_uu = Inf * ones(model_ft.m)
_uu[model_ft.idx_u] .= Inf
_uu[end] = 5.0 * h

_ul = zeros(model_ft.m)
_ul[model_ft.idx_u] .= -Inf
_ul[end] = 0.5 * h
ul, uu = control_bounds(model_ft, T, _ul, _uu)

xl, xu = state_bounds(model_ft, T,
    x1 = [q1; q1],
    xT = [Inf * ones(nq); qT])

# Objective
include_objective(["velocity", "nonlinear_stage"])

x0 = configuration_to_state(q_ref)

# penalty on slack variable
obj_penalty = PenaltyObjective(1.0e4, model_ft.m - 1)

# quadratic tracking objective
# Σ (x - xref)' Q (x - x_ref) + (u - u_ref)' R (u - u_ref)
obj_control = quadratic_time_tracking_objective(
    [Diagonal(0.0 * ones(2 * model.nq)) for t = 1:T],
    [Diagonal([0.0 * ones(model_ft.nu)..., 0.0 * ones(model_ft.nc + model_ft.nb)..., zeros(model_ft.m - model_ft.nu - model_ft.nc - model_ft.nb)...]) for t = 1:T-1],
    [deepcopy(x0[t]) for t = 1:T],
    [zeros(model_ft.m) for t = 1:T],
    1.0)

# quadratic velocity penalty
# Σ v' Q v
obj_velocity = velocity_objective(
    [Diagonal(1.0 * ones(model_ft.nq)) for t = 1:T-1],
    model_ft.nq,
    h = h,
    idx_angle = collect([3, 4, 5, 6, 7, 8, 9, 10, 11]))


t_h = kinematics_1(model_ft, qM, body = :torso, mode = :com)[2]
θ_range = range(0.5 * π, stop = 0.5 * π + 2 * π, length = T)
function l_stage(x, u, t)
	J = 0.0

	q = view(x, 1:11)

	J += 10000.0 * (kinematics_1(model_ft, q, body = :torso, mode = :com)[2] - kinematics_1(model_ft, q_ref[t], body = :torso, mode = :com)[2])^2.0
	# J += 10000.0 * (q[3] - θ_range[t])^2.0

	return J
end

l_stage(x) = l_stage(x, nothing, T)
obj_stage = nonlinear_stage_objective(l_stage, l_stage)

obj = MultiObjective([obj_penalty,
                      obj_control,
                      obj_velocity,
                      obj_stage])


# Constraints
include_constraints(["stage", "contact", "free_time"])
con_contact = contact_constraints(model_ft, T)
con_free_time = free_time_constraints(T)

pr1_1 = kinematics_2(model, q1, body = :calf_1, mode = :ee)
pr2_1 = kinematics_2(model, q1, body = :calf_2, mode = :ee)

pf1_1 = kinematics_3(model, q1, body = :calf_3, mode = :ee)
pf2_1 = kinematics_3(model, q1, body = :calf_4, mode = :ee)

pr1_T = kinematics_2(model, qT, body = :calf_1, mode = :ee)
pr2_T = kinematics_2(model, qT, body = :calf_2, mode = :ee)

pf1_T = kinematics_3(model, qT, body = :calf_3, mode = :ee)
pf2_T = kinematics_3(model, qT, body = :calf_4, mode = :ee)


function pinned_1!(c, x, u, t)
    q = view(x, 1:11)
    c[1:2] = pr1_1 - kinematics_2(model, q, body = :calf_1, mode = :ee)
    c[3:4] = pf1_1 - kinematics_3(model, q, body = :calf_3, mode = :ee)
	c[5:6] = pr2_1 - kinematics_2(model, q, body = :calf_2, mode = :ee)
    c[7:8] = pf2_1 - kinematics_3(model, q, body = :calf_4, mode = :ee)
    nothing
end

function pinned_T!(c, x, u, t)
    q = view(x, 11 .+ (1:11))
    c[1:2] = pr1_T - kinematics_2(model, q, body = :calf_1, mode = :ee)
    c[3:4] = pf1_T - kinematics_3(model, q, body = :calf_3, mode = :ee)
	c[5:6] = pr2_T - kinematics_2(model, q, body = :calf_2, mode = :ee)
    c[7:8] = pf2_T - kinematics_3(model, q, body = :calf_4, mode = :ee)
    nothing
end

n_stage = 8
t_idx1 = vcat([t for t = 1:3]...)
t_idxT = vcat([t for t = T-2:T]...)
con_pinned1 = stage_constraints(pinned_1!, n_stage, (1:0), t_idx1)
con_pinnedT = stage_constraints(pinned_T!, n_stage, (1:0), t_idxT)

con = multiple_constraints([con_contact, con_free_time, con_pinned1, con_pinnedT])

# Problem
prob = trajectory_optimization_problem(model_ft,
               obj,
               T,
               # h = h,
               xl = xl,
               xu = xu,
               ul = ul,
               uu = uu,
               con = con)

# trajectory initialization
u0 = [[1.0e-3 * rand(model_ft.m - 1); h] for t = 1:T-1] # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob)
# z0 .+= 1.0e-3 * randn(prob.num_var)

# Solve
# include_snopt()

@time z̄ , info = solve(prob, copy(z0),
    # nlp = :SNOPT7,
    tol = 1.0e-2, c_tol = 1.0e-2,
	max_iter = 2000,
    time_limit = 60 * 3, mapl = 5)

check_slack(z̄, prob)
x̄, ū = unpack(z̄, prob)
tf, t, h̄ = get_time(ū)

# @save joinpath(pwd(), "examples/trajectories/quadruped_backflip.jld2") x̄ ū h̄

# Visualize
visualize!(vis, model_ft, state_to_configuration(x̄), Δt = h̄[1])

visualize!(vis, model_ft,
	[[state_to_configuration(x̄)[1] for i = 1:10]..., state_to_configuration(x̄)..., [state_to_configuration(x̄)[end] for i = 1:10]...],
	Δt = ū[1][end])

# using Plots
# plot(t[1:end-1], hcat(ū...)[model_ft.idx_u, :]', linetype = :steppost,
# 	width = 2.0, label = "", xlabel= "time (s)", ylabel = "control")

# save trajectory
qm = state_to_configuration(x̄)
um = [u[model.idx_u] for u in ū]
γm = [u[model.idx_λ] for u in ū]
bm = [u[model.idx_b] for u in ū]
ψm = [u[model.idx_ψ] for u in ū]
ηm = [u[model.idx_η] for u in ū]
hm = mean(h̄)
μm = model_ft.μ
@save joinpath(@__DIR__, "quadruped_jump_v2.jld2") qm um γm bm ψm ηm μm hm

using Plots
plot(hcat(γm...)', linetype = :steppost)
plot(hcat(bm...)', linetype = :steppost)
