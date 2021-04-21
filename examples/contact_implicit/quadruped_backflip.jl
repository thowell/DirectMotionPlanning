# Model
include_model("quadruped")
model_ft = free_time_model(model)

# Horizon
T = 25

# Time step
tf = 0.75
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
qM[2] += 0.5
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
qT[1] -= model_ft.l_torso
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

q_ref = [linear_interpolation(q1, qM, 13)...,
    linear_interpolation(qM, qT, 13)...]

visualize!(vis, model_ft, q_ref)

# Bounds

# control
# u = (τ1..4, λ1..2, β1..4, ψ1..2, η1...4, s1)
# ul <= u <= uu
_uu = Inf * ones(model_ft.m)
_uu[model_ft.idx_u] .= Inf
_uu[end] = 2.0 * h

_ul = zeros(model_ft.m)
_ul[model_ft.idx_u] .= -Inf
_ul[end] = 0.5 * h
ul, uu = control_bounds(model_ft, T, _ul, _uu)

xl, xu = state_bounds(model_ft, T,
    x1 = [q1; q1],
    xT = [qT; qT])

# Objective
include_objective(["velocity", "nonlinear_stage"])

x0 = configuration_to_state(q_ref)

# penalty on slack variable
obj_penalty = PenaltyObjective(1.0e4, model_ft.m - 1)

# quadratic tracking objective
# Σ (x - xref)' Q (x - x_ref) + (u - u_ref)' R (u - u_ref)
obj_control = quadratic_time_tracking_objective(
    [Diagonal(ones(2 * model.nq)) for t = 1:T],
    [Diagonal([1.0 * ones(model_ft.nu)..., zeros(model_ft.m - model_ft.nu)...]) for t = 1:T-1],
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

	J += 10000.0 * (kinematics_1(model_ft, q, body = :torso, mode = :com)[2] - t_h)^2.0
	# J += 10000.0 * (q[3] - θ_range[t])^2.0

	return J
end

l_stage(x) = l_stage(x, nothing, T)
obj_stage = nonlinear_stage_objective(l_stage, l_stage)
# #
# # # torso lateral
# # l_stage_torso_lat(x, u, t) = (1.0 * (kinematics_1(model_ft, view(x, q2_idx), body = :torso, mode = :com)[1] - kinematics_1(model_ft, view(x0[t], q2_idx), body = :torso, mode = :com)[1])^2.0)
# # l_terminal_torso_lat(x) = (0.0 * (kinematics_1(model_ft, view(x, q2_idx), body = :torso, mode = :com)[1] - kinematics_1(model_ft, view(x0[T], q2_idx), body = :torso, mode = :com)[1])^2.0)
# # obj_torso_lat = nonlinear_stage_objective(l_stage_torso_lat, l_terminal_torso_lat)
# #
# # foot 1 height
# l_stage_fh1(x, u, t) = 1000.0 * (kinematics_2(model_ft, view(x, q2_idx), body = :calf_1, mode = :ee)[2] - 0.5)^2.0
# l_terminal_fh1(x) = 0.0 * (kinematics_2(model_ft, view(x, q2_idx), body = :calf_1, mode = :ee)[2])^2.0
# obj_fh1 = nonlinear_stage_objective(l_stage_fh1, l_terminal_fh1)
#
# # foot 2 height
# l_stage_fh2(x, u, t) = 1000.0 * (kinematics_2(model_ft, view(x, q2_idx), body = :calf_2, mode = :ee)[2] - 0.5)^2.0
# l_terminal_fh2(x) = 0.0 * (kinematics_2(model_ft, view(x, q2_idx), body = :calf_2, mode = :ee)[2])^2.0
# obj_fh2 = nonlinear_stage_objective(l_stage_fh2, l_terminal_fh2)
#
# # foot 3 height
# l_stage_fh3(x, u, t) = 100.0 * (kinematics_3(model_ft, view(x, q2_idx), body = :calf_3, mode = :ee)[2] - 0.5)^2.0
# l_terminal_fh3(x) = 0.0 * (kinematics_3(model_ft, view(x, q2_idx), body = :calf_3, mode = :ee)[2])^2.0
# obj_fh3 = nonlinear_stage_objective(l_stage_fh3, l_terminal_fh3)
#
# # foot 4 height
# l_stage_fh4(x, u, t) = 100.0 * (kinematics_3(model_ft, view(x, q2_idx), body = :calf_4, mode = :ee)[2] - 0.5)^2.0
# l_terminal_fh4(x) = 0.0 * (kinematics_3(model_ft, view(x, q2_idx), body = :calf_4, mode = :ee)[2])^2.0
# obj_fh4 = nonlinear_stage_objective(l_stage_fh4, l_terminal_fh4)

obj = MultiObjective([obj_penalty,#,
                      # obj_control,
                      obj_velocity,
                      obj_stage])
                      # # obj_torso_lat,
                      # obj_fh1,
                      # obj_fh2,
                      # obj_fh3,
                      # obj_fh4])

# Constraints
include_constraints(["contact", "free_time"])
con_contact = contact_constraints(model_ft, T)
con_free_time = free_time_constraints(T)
con = multiple_constraints([con_contact, con_free_time])

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

using Plots
plot(t[1:end-1], hcat(ū...)[model_ft.idx_u, :]', linetype = :steppost,
	width = 2.0, label = "", xlabel= "time (s)", ylabel = "control")
