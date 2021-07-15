using Plots

# Model
include_model("planar_push_block_v2")

# Horizon
T = 26

# Time step
tf = 2.5
h = tf / (T-1)


# Bounds
_uu = Inf * ones(model.m)
_uu[model.idx_u] .= Inf
_uu[model.idx_u[1:2:8]] .= 1.0
_uu[model.idx_u[2:2:9]] .= 0.5 * π
_uu[model.idx_λ] .= 1.0
_ul = zeros(model.m)
_ul[model.idx_u] .= -Inf
_ul[model.idx_u[1:2:8]] .= 0.0
_ul[model.idx_u[2:2:9]] .= -0.5 * π
_ul[model.idx_λ] .= 1.0

ul, uu = control_bounds(model, T, _ul, _uu)

# Initial and final states
q1 = [0.0, 0.0, 0.0]
x1 = [q1; q1]
qT = [1.0; 0.0; 0.0]
xT = [qT; qT]
xl, xu = state_bounds(model, T, x1 = x1, xT = xT)

# Objective
include_objective(["velocity", "nonlinear_stage"])
obj_velocity = velocity_objective(
    [t > T / 2 ? Diagonal(10.0 * ones(model.nq)) : Diagonal(1.0 * ones(model.nq)) for t = 1:T-1],
    model.nq,
    h = h,
    idx_angle = collect([3]))

obj_tracking = quadratic_tracking_objective(
    [Diagonal(1.0 * ones(model.n)) for t = 1:T],
    # [Diagonal(0.1 * ones(model.m)) for t = 1:T-1],
	[Diagonal([0.1 * ones(8);
		zeros(4);
		zeros(model.nc);
		ones(model.nb);
		zeros(model.m - model.nu - model.nc - model.nb)]) for t = 1:T-1],
    [xT for t = 1:T],
    [zeros(model.m) for t = 1:T])

obj_penalty = PenaltyObjective(1.0e6, model.m)

function cost_l1(x, u, t)
	J = 0.0

	if t < T
		J += 1000.0 * sum(u[model.idx_u][9:12])
	end

	return J
end

cost_l1(x) = cost_l1(x, nothing, T)
obj_l1 = nonlinear_stage_objective(cost_l1, cost_l1)

obj = MultiObjective([obj_tracking, obj_penalty, obj_velocity, obj_l1])

# Constraints
include_constraints(["contact", "stage"])
t_idx = vcat([t for t = 1:T-1])

function con_l1!(c, x, u, t)

	u_f = u[model.idx_u][1:2:8]
	u_l1 = u[model.idx_u][9:12]

	c[1:4] = u_l1 - u_f
	c[5:8] = u_f + u_l1

    nothing
end

n_l1 = 8
con_l1 = stage_constraints(con_l1!, n_l1, (1:8), t_idx)

# function control_limits_con!(c, x, u, t)
# 	q = x[nq .+ (1:nq)]
# 	θ = q[3]
# 	R = rotation_matrix(θ)
# 	u_ctrl = u[model.idx_u]
#
# 	c[1] = -1.0 * (R' * u_ctrl[1:2])[1]
# 	c[2] = (model.μ[end] * u_ctrl[1])^2.0 - u_ctrl[2]^2.0
#
# 	c[3] = -1.0 * (R' * u_ctrl[3:4])[2]
# 	c[4] = (model.μ[end] * u_ctrl[4])^2.0 - u_ctrl[3]^2.0
#
# 	c[5] = (R' * u_ctrl[5:6])[1]
# 	c[6] = (model.μ[end] * u_ctrl[5])^2.0 - u_ctrl[6]^2.0
#
# 	c[7] = (R' * u_ctrl[7:8])[2]
# 	c[8] = (model.μ[end] * u_ctrl[8])^2.0 - u_ctrl[7]^2.0
# end
#
# n_ctrl_lim = 8
# con_ctrl_lim = stage_constraints(control_limits_con!, n_ctrl_lim, (1:8), t_idx)

con_contact = contact_constraints(model, T)
con = multiple_constraints([con_contact, con_l1])#, con_ctrl_lim])

# Problem
prob = trajectory_optimization_problem(model,
               obj,
               T,
               h = h,
               xl = xl,
               xu = xu,
               ul = ul,
               uu = uu,
               con = con)

# Trajectory initialization
x0 = linear_interpolation(x1, xT, T) # linear interpolation on state
u0 = [0.001 * rand(model.m) for t = 1:T-1] # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem
@time z̄, info = solve(prob, copy(z0), tol = 1.0e-3, c_tol = 1.0e-3)

check_slack(z̄, prob)
x̄, ū = unpack(z̄, prob)
q̄ = state_to_configuration(x̄)

q = state_to_configuration(x̄)
u = [u[model.idx_u] for u in ū]
γ = [u[model.idx_λ] for u in ū]
b = [u[model.idx_b] for u in ū]
h̄ = h

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
#open(vis)
visualize!(vis, model,
    q̄, ū,
    Δt = h)

# settransform!(vis["/Cameras/default"],
# 	compose(Translation(0.0, 0.0, 50.0), LinearMap(RotZ(0.5 * pi) * RotY(-pi/2.5))))
# setprop!(vis["/Cameras/default/rotated/<object>"], "zoom", 25)

plot(hcat(q̄...)', color = :red, width = 1.0, labels = "")
plot(hcat([ū..., ū[end]]...)[model.idx_u[1:2:8], :]', linetype = :steppost, color = :red, width = 1.0, labels = "impulse magnitude")

[ū[t][model.idx_u[1:2:8]] for t = 1:T-1]
