# Model
include_model("hopper")
model_ft = free_time_model(model)

# Horizon
T = 101

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
    [Diagonal([1.0, 1.0, zeros(model_ft.m - model_ft.nu - 1)..., 0.0]) for t = 1:T-1],
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

optimize = true
if optimize
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

using Plots
plot(hcat(ū...)[1:2, :]',
    linetype = :steppost,
    label = "",
    color = :red,
    width = 2.0)

plot!(hcat(u_proj...)[1:2, :]',
    linetype = :steppost,
    label = "", color = :black)

plot(hcat(state_to_configuration(x̄)...)',
    color = :red,
    width = 2.0,
	label = "")

plot!(hcat(state_to_configuration(x_proj)...)',
    color = :black,
	label = "")

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
visualize!(vis, model_ft, state_to_configuration(x_proj), Δt = h̄[1])

# Reference trajectory
x_track = deepcopy(x_proj)
u_track = deepcopy(u_proj)
for i = 1:5
	x_track = [x_track..., x_track[2:T]...]
	u_track = [u_track..., u_track[1:T-1]...]
end

T_track = length(x_track)

# x_shift = 0.25
# x_track_shift = deepcopy(x_track)
# for t = 1:T_track
#     x_track_shift[t] = x_track_shift[t] + [x_shift; 0.0; 0.0; 0.0; x_shift; 0.0; 0.0; 0.0]
# end
# x_track_shift
plot(hcat(state_to_configuration(x_track)...)',
    color = :black,
	label = "")

# plot(hcat(state_to_configuration(x_track_shift)...)',
#     color = :black,
# 	label = "")

plot(hcat(u_track...)',
	color = :black,
	label = "")

K, P = tvlqr(model_ft, x_track, u_track, h̄[1],
	[Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0]) for t = 1:T_track],
	[Diagonal(1.0 * ones(model_ft.m)) for t = 1:T_track - 1])

# K_shift, P_shift = tvlqr(model_ft, x_track_shift, u_track, h̄[1],
# 	[Diagonal(1.0 * ones(model_ft.n)) for t = 1:T_track],
# 	[Diagonal(1.0 * ones(model_ft.m)) for t = 1:T_track - 1])

K_vec = [vec(K[t]) for t = 1:T_track-1]
P_vec = [vec(P[t]) for t = 1:T_track-1]

plot(hcat(K_vec...)', label = "")
plot(hcat(P_vec...)', label = "")

include(joinpath(pwd(), "src/contact_simulator/simulator_variable_time_step.jl"))
model_sim = model

T_sim = 1 * T_track
tf_track = h̄[1] * (T_track - 1)
t_sim = range(0, stop = tf_track, length = T_sim)
t_ctrl = range(0, stop = tf_track, length = T_track)
h_sim = tf_track / (T_sim - 1)
x_track_stack = hcat(x_track...)

x_sim = [copy(x_proj[1])]
u_sim = []
T_horizon = T_sim - T
x_shift = 0.0
x_shift_vec = [x_shift; 0.0; 0.0; 0.0; x_shift; 0.0; 0.0; 0.0]

for tt = 1:T_horizon-1
	t = t_sim[tt]
	i = searchsortedlast(t_ctrl, t)
	println("t: $t")
	println("	i: $i")
	# ii = searchsortedlast(t_sim, t)

	# x_cubic = zeros(model_sim.n)
	# for j = 1:model_sim.n
	# 	interp_cubic = CubicSplineInterpolation(t_ctrl, x_track_stack[j, :])
	# 	x_cubic[j] = interp_cubic(t)
	# end

	# push!(u_sim, u_track[i][1:end-1] - K[i] * (x_sim[end] - x_track[i]))

	w0 = (tt == 101 ? 1000.0 * [1.0; 0.0; 0.0; 0.0] : 1.0e-5 * randn(model.nq))

	_q_sim = [x_sim[end][1:nq], x_sim[end][nq .+ (1:nq)]]
	_v_sim = [(x_sim[end][nq .+ (1:nq)] - x_sim[end][1:nq]) / h̄[1]]

	d = 3
	_h = h̄[1] / convert(Float64, d)
	for p = 1:d

		# x_cubic = zeros(model_sim.n)
		# for j = 1:model_sim.n
		# 	interp_cubic = CubicSplineInterpolation(t_ctrl, x_track_stack[j, :])
		# 	x_cubic[j] = interp_cubic(t + p * _h)
		# end

		ii = max(1, searchsortedlast(t_sim, t - h̄[1] +  p * _h))
		_x = [x_sim[ii][(1:nq)]; _q_sim[end]]

		_q = step_contact(model,
			_v_sim[end], _q_sim[end-1], _q_sim[end],
			(u_track[i][1:end-1] - K[i] * (_x - (x_track[i] + 0.0 * x_shift_vec)))[1:2],
			# u_sim[end][1:2],
			p == 1 ? 0.0 * w0 : 0.0 * randn(model.nq),
			_h)
		_v = (_q - _q_sim[end]) / _h

		push!(_q_sim, _q)
		push!(_v_sim, _v)
	end

	push!(x_sim, [x_sim[end][nq .+ (1:nq)]; _q_sim[end]])
end

plot(hcat(state_to_configuration(x_track[1:1:T_horizon])...)',
    labels = "", legend = :bottomleft,
    width = 2.0, color = ["red" "green" "blue" "orange"], linestyle = :dash)

plot!(hcat(state_to_configuration(x_sim[1:1:T_horizon])...)',
    labels = "", legend = :bottom,
    width = 1.0, color = ["red" "green" "blue" "orange"])

# plot(hcat(u_track...)[1:2, :]',
# 	width = 2.0,
# 	linetype = :steppost,
# 	linestyle = :dash)
#
# plot!(hcat(u_sim...)[1:2, :]',
# 	width = 1.0,
# 	linetype = :steppost)
#
# plot(hcat(u_track...)[3:5, 100:200]',
# 	width = 2.0,
# 	color = ["red" "green" "blue"],
# 	linetype = :steppost,
# 	linestyle = :dash)
#
# plot!(hcat(u_sim...)[3:5, 100:200]',
# 	width = 1.0,
# 	color = ["red" "green" "blue"],
# 	linetype = :steppost)

vis = Visualizer()
render(vis)
visualize!(vis, model_ft, state_to_configuration(x_sim), Δt = h̄[1])

# # Simulator
# function dynamics(model::Hopper, v1, q1, q2, q3, u, λ, b, w, h)
# 	(M_func(model, q1) * v1
# 	- M_func(model, q2) * (SVector{4}(q3) - SVector{4}(q2)) / h
# 	+ h * (transpose(B_func(model, q3)) * SVector{2}(u)
# 	+ transpose(N_func(model, q3)) * SVector{1}(λ)
# 	+ transpose(P_func(model, q3)) * SVector{2}(b)
# 	- G_func(model, q2)))
# end
#
# function maximum_dissipation(model::Hopper, q2, q3, ψ, η, h)
# 	ψ_stack = ψ[1] * ones(model.nb)
# 	return P_func(model, q3) * (q3 - q2) / h + ψ_stack - η
# end
#
# function friction_cone(model::Hopper, λ, b)
# 	return @SVector [model.μ * λ[1] - sum(b)]
# end
#
# mutable struct MOISimulator <: MOI.AbstractNLPEvaluator
#     num_var::Int                 # number of decision variables
#     num_con::Int                 # number of constraints
#     primal_bounds
#     constraint_bounds
#
# 	model
# 	slack_penalty
# 	v1
# 	q1
# 	q2
# 	u
# 	w
# 	h
# end
#
# function simulator_problem(model, v1, q1, q2, u, w, h; slack_penalty = 1.0e5)
# 	num_var = model.nq + model.nc + model.nb + model.nc + model.nb + model.ns
# 	num_con = model.nq + model.nc + model.nc + model.nb + 3
#
# 	zl = zeros(num_var)
# 	zl[1:model.nq] .= -Inf
# 	zu = Inf * ones(num_var)
#
# 	cl = zeros(num_con)
# 	cu = zeros(num_con)
# 	cu[model.nq .+ (1:model.nc)] .= Inf
# 	cu[model.nq + model.nc .+ (1:model.nc)] .= Inf
# 	cu[model.nq + model.nc + model.nc + model.nb + 1] = Inf
# 	cu[model.nq + model.nc + model.nc + model.nb + 2] = Inf
# 	cu[model.nq + model.nc + model.nc + model.nb + 3] = Inf
#
# 	MOISimulator(num_var,
# 		num_con,
# 		(zl, zu),
# 		(cl, cu),
# 		model,
# 		slack_penalty,
# 		v1,
# 		q1,
# 		q2,
# 		u,
# 		w,
# 		h)
# end
# prob_sim = simulator_problem(model, ones(nq), ones(nq), ones(nq), zeros(nu), ones(nq), h)
# z0 = rand(prob_sim.num_var)
#
# function MOI.eval_objective(prob::MOISimulator, x)
#     return prob.slack_penalty * x[prob.num_var]
# end
#
# MOI.eval_objective(prob_sim, z0)
#
# function MOI.eval_objective_gradient(prob::MOISimulator, grad_f, x)
#     grad_f .= 0.0
# 	grad_f[prob.num_var] = prob.slack_penalty
# 	return nothing
# end
#
# ∇obj = zeros(prob_sim.num_var)
# MOI.eval_objective_gradient(prob_sim, ∇obj, z0)
#
# function MOI.eval_constraint(prob::MOISimulator, g, x)
# 	model = prob.model
# 	q3 = view(x, 1:model.nq)
# 	λ = view(x, model.nq .+ (1:model.nc))
# 	b = view(x, model.nq + model.nc .+ (1:model.nb))
# 	ψ = view(x, model.nq + model.nc + model.nb .+ (1:model.nc))
# 	η = view(x, model.nq + model.nc + model.nb + model.nc .+ (1:model.nb))
# 	s = view(x, model.nq + model.nc + model.nb + model.nc + model.nb .+ (1:model.ns))
# 	# λ = 0
# 	# b = 0
#     g[1:model.nq] = dynamics(model,
# 		prob.v1, prob.q1, prob.q2, q3, prob.u, λ, b, prob.w, prob.h)
# 	g[model.nq .+ (1:model.nc)] = ϕ_func(model, q3)
# 	g[model.nq + model.nc .+ (1:model.nc)] = friction_cone(model, λ, b)
# 	g[model.nq + model.nc + model.nc .+ (1:model.nb)] = maximum_dissipation(model,
# 		prob.q2, q3, ψ, η, prob.h)
# 	g[model.nq + model.nc + model.nc + model.nb + 1] = s[1] - (λ' * ϕ_func(model, q3))[1]
# 	g[model.nq + model.nc + model.nc + model.nb + 2] = s[1] - (ψ' * friction_cone(model, λ, b))[1]
# 	g[model.nq + model.nc + model.nc + model.nb + 3] = s[1] - (η' * b)[1]
#
#     return nothing
# end
# prob_sim.h
# c0 = zeros(prob_sim.num_con)
# MOI.eval_constraint(prob_sim, c0, z0)
#
# function MOI.eval_constraint_jacobian(prob::MOISimulator, jac, x)
#     con!(g, z) = MOI.eval_constraint(prob, g, z)
# 	ForwardDiff.jacobian!(reshape(jac, prob.num_var, prob.num_con), con!, zeros(prob.num_con), x)
#     return nothing
# end
#
# ∇c0 = vec(zeros(prob_sim.num_con, prob_sim.num_var))
# MOI.eval_constraint_jacobian(prob_sim, ∇c0, z0)
#
# function sparsity_jacobian(prob::MOISimulator)
#     row = []
#     col = []
#
#     r = (1:prob.num_con)
#     c = (1:prob.num_var)
#
#     row_col!(row, col, r, c)
#
#     return collect(zip(row, col))
# end
#
# sparsity_jacobian(prob_sim)
#
# @time solve(prob_sim, z0)
#
# function step_contact(model, v1, q1, q2, u, w, h)
#     prob = simulator_problem(model, v1, q1, q2, u, w, h)
#     z0 = [copy(q2); 1.0e-5 * ones(model.nc + model.nb + model.nc + model.nb + model.ns)]
#     @time z , info = solve(prob, copy(z0), tol = 1.0e-8, c_tol = 1.0e-8, mapl = 0)
#
#     @assert z[end] < 1.0e-8
#
#     return z[1:model.nq]
# end
#
q_proj = state_to_configuration(x_proj)
q_sim = [q_proj[1], q_proj[2]]
v_sim = [(q_proj[2] - q_proj[1]) / h̄[1]]

for t = 1:T-1
	# x rate
	_q_sim = [q_sim[end-1], q_sim[end]]
	_v_sim = [v_sim[end]]

	d = 10
	_h = h̄[1] / convert(Float64, d)
	for i = 1:d
		_q = step_contact(model,
			_v_sim[end], _q_sim[end-1], _q_sim[end], u_proj[t][model.idx_u], zeros(model.d), _h)
		_v = (_q - _q_sim[end]) / _h

		push!(_q_sim, _q)
		push!(_v_sim, _v)
	end

	push!(q_sim, _q_sim[end])
	push!(v_sim, _v_sim[end])
end

plot(hcat(q_proj...)',
    color = :red,
    width = 2.0,
	label = "")
plot!(hcat(q_sim...)',
    color = :black,
    width = 1.0,
	label = "")

plot(hcat(v_sim...)',
    color = :black,
    width = 1.0,
	label = "")
