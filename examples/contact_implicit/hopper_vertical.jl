# Model
include_model("hopper")
model_ft = free_time_model(model)

# Horizon
T = 51

# Time step
tf = 1.0
h = tf / (T - 1)

# Bounds
_uu = Inf * ones(model_ft.m)
_uu[model_ft.idx_u] = model_ft.uU
_uu[end] = 2.0 * h
_ul = zeros(model_ft.m)
_ul[model_ft.idx_u] = model_ft.uL
_ul[end] = 0.5 * h
ul, uu = control_bounds(model_ft, T, _ul, _uu)

# Initial and final states
z_h = 0.25
q1 = [0.0, 0.5 + z_h, 0.0, 0.25]

xl, xu = state_bounds(model_ft, T,
		[model_ft.qL; model_ft.qL],
		[model_ft.qU; model_ft.qU],
        x1 = [q1; Inf * ones(model.nq)])

# Objective
include_objective("velocity")
obj_tracking = quadratic_time_tracking_objective(
    [Diagonal(zeros(model_ft.n)) for t = 1:T],
    [Diagonal([1.0, 1.0,
		ones(model_ft.nc)..., ones(model_ft.nb)...,
		zeros(model_ft.m - model_ft.nu - model_ft.nc - model_ft.nb - 1)..., 0.0])
		for t = 1:T-1],
    [zeros(model_ft.n) for t = 1:T],
    [zeros(model_ft.m) for t = 1:T],
    1.0)
obj_contact_penalty = PenaltyObjective(1.0e5, model_ft.m - 1)
obj_velocity = velocity_objective(
    [Diagonal(1.0 * ones(model_ft.nq)) for t = 1:T-1],
    model_ft.nq,
    h = h,
    idx_angle = collect([3]))
obj = MultiObjective([obj_tracking, obj_contact_penalty, obj_velocity])

# Constraints
include_constraints(["free_time", "contact", "loop"])
con_free_time = free_time_constraints(T)
con_contact = contact_constraints(model_ft, T)
con_loop = loop_constraints(model, (1:model.n), 1, T)
con = multiple_constraints([con_free_time, con_contact, con_loop])

# Problem
prob = trajectory_optimization_problem(model_ft,
               obj,
               T,
               xl = xl,
               xu = xu,
               ul = ul,
               uu = uu,
               con = con)

# Trajectory initialization
x0 = [[q1; q1] for t = 1:T] # linear interpolation on state
u0 = [[1.0e-3 * rand(model_ft.m-1); h] for t = 1:T-1] # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem
@time z̄, info = solve(prob, copy(z0),
	nlp = :ipopt,
	tol = 1.0e-7, c_tol = 1.0e-7, mapl = 5,
	time_limit = 60)
@show check_slack(z̄, prob)
x̄, ū = unpack(z̄, prob)
tf, t, h̄ = get_time(ū)

q = state_to_configuration(x̄)
u = [u[model.idx_u] for u in ū]
γ = [u[model.idx_λ] for u in ū]
b = [u[model.idx_b] for u in ū]
h̄ = mean(h̄)
@save joinpath(pwd(), "examples/trajectories/hopper_vertical_gait.jld2") z̄ x̄ ū h̄ q u γ b

if true
	# include_snopt()
	@time z̄ , info = solve(prob, copy(z0),
		nlp = :ipopt,
		tol = 1.0e-5, c_tol = 1.0e-5, mapl = 5,
		time_limit = 60)
	@show check_slack(z̄, prob)
	x̄, ū = unpack(z̄, prob)
	tf, t, h̄ = get_time(ū)

	# projection
	Q = [Diagonal(ones(model_ft.n)) for t = 1:T]
	R = [Diagonal(0.1 * ones(model_ft.m)) for t = 1:T-1]
	x_proj, u_proj = lqr_projection(model_ft, x̄, ū, h̄[1], Q, R)

	@show tf
	@show h̄[1]
	@save joinpath(pwd(), "examples/trajectories/hopper_vertical_gait.jld2") x̄ ū h̄ x_proj u_proj
else
	@load joinpath(pwd(), "examples/trajectories/hopper_vertical_gait.jld2") x̄ ū h̄ x_proj u_proj
end

z_traj = [x[2] for x in x̄]
k_traj = [u[3] for u in ū]
λ_traj = [u[model_ft.idx_λ] for u in ū]
f_traj = [u[2] for u in ū]

plot(hcat(k_traj...)', linetype = :steppost, label = "impedance")
plot!(h̄[1] * hcat(λ_traj...)', linetype = :steppost, label = ["" "λ1"])
plot!(h̄[1] * hcat(f_traj...)', linetype = :steppost, label = "force")

plot(hcat(z_traj...)', linetype = :steppost, label = "z")


#
# using Plots
# plot(hcat(ū...)[1:2, :]',
#     linetype = :steppost,
#     label = "",
#     color = :red,
#     width = 2.0)
#
# plot!(hcat(u_proj...)[1:2, :]',
#     linetype = :steppost,
#     label = "", color = :black)
#
# plot(hcat(state_to_configuration(x̄)...)',
#     color = :red,
#     width = 2.0,
# 	label = "")
#
# plot!(hcat(state_to_configuration(x_proj)...)',
#     color = :black,
# 	label = "")
include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
visualize!(vis, model_ft,
	state_to_configuration([x̄..., ([x̄[2:end] for i = 1:5]...)...]),
	Δt = h̄[1],
	scenario = :vertical)

# @time z̄ , info = solve(prob, copy(z̄),
# 	nlp = :ipopt,
# 	tol = 1.0e-2, c_tol = 1.0e-2, mapl = 0,
# 	time_limit = 60)
#
# obj_track = quadratic_tracking_objective(
#     [Diagonal(100.0 * ones(model.n)) for t = 1:T],
#     [Diagonal(ones(model.m)) for t = 1:T-1],
#     [x̄[t] for t = 1:T],
#     [ū[t][1:end-1] for t = 1:T-1])
#
#
# prob_track = trajectory_optimization_problem(model,
#                obj_track,
#                T,
#                xl = xl,
#                xu = xu,
#                ul = [ul[t][1:end-1] for t = 1:T-1],
#                uu = [uu[t][1:end-1] for t = 1:T-1],
#                con = con_contact)
#
# z_track = pack(x̄, [ū[t][1:end-1] for t = 1:T-1], prob_track) #+ 1.0e-2 * ones(prob_track.num_var)
# # z_track[1:model.n] .+= 0.25
# # prob_track.primal_bounds[1][1:model.n] = z_track[1:model.n]
# # prob_track.primal_bounds[2][1:model.n] = z_track[1:model.n]
#
# @time z_sol , info = solve(prob_track, copy(z_track),
# 	nlp = :ipopt,
# 	tol = 1.0e-5, c_tol = 1.0e-5, mapl = 0,
# 	time_limit = 60)
#
# x_sol, u_sol = unpack(z_sol, prob_track)
# plot(hcat(state_to_configuration(x̄)...)')
# plot!(hcat(state_to_configuration(x_sol)...)')
#
# plot(hcat([ū[t][1:end-1] for t = 1:T-1]...)', color = :red, width = 2.0, label = "")
# plot!(hcat(u_sol...)', color = :black, width = 1.0)
#
# x_sol[1] - z_track[1:model.n]
