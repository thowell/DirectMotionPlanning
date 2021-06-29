using Plots

# Model
include_model("planar_push_block")

# Horizon
T = 26

# Time step
tf = 2.5
h = tf / (T-1)


# Bounds
_uu = Inf * ones(model.m)
_uu[model.idx_u] .= Inf
_uu[model.idx_u[1:8]] .= 5.0
_uu[model.idx_λ] .= 1.0
_ul = zeros(model.m)
_ul[model.idx_u] .= -Inf
_ul[model.idx_u[1:8]] .= -5.0
_ul[model.idx_λ] .= 1.0

ul, uu = control_bounds(model, T, _ul, _uu)

# Initial and final states
q1 = [0.0, 0.0, 0.0]
x1 = [q1; q1]
qT = [1.0; 1.0; 0.5 * π]
xT = [qT; qT]
xl, xu = state_bounds(model, T, x1 = x1, xT = xT)

# Objective
include_objective(["velocity", "control_velocity"])
obj_velocity = velocity_objective(
    [t > T / 2 ? Diagonal(10.0 * ones(model.nq)) : Diagonal(10.0 * ones(model.nq)) for t = 1:T-1],
    model.nq,
    h = h,
    idx_angle = collect([3]))

obj_tracking = quadratic_tracking_objective(
    [Diagonal(1.0 * ones(model.n)) for t = 1:T],
    # [Diagonal(0.1 * ones(model.m)) for t = 1:T-1],
	[Diagonal([0.1 * ones(model.nu);
		zeros(model.nc);
		ones(model.nb);
		zeros(model.m - model.nu - model.nc - model.nb)]) for t = 1:T-1],
    [xT for t = 1:T],
    [zeros(model.m) for t = 1:T])

obj_ctrl_vel = control_velocity_objective(Diagonal([1.0 * ones(model.nu); 0.0 * ones(model.nc + model.nb); zeros(model.m - model.nu - model.nc - model.nb)]))

obj_penalty = PenaltyObjective(1.0e4, model.m)
obj = MultiObjective([obj_tracking, obj_penalty, obj_velocity, obj_ctrl_vel])

# Constraints
include_constraints(["contact", "stage"])
t_idx = vcat([t for t = 1:T-1])

function control_comp_con!(c, x, u, t)
	q = x[nq .+ (1:nq)]
	s = u[model.idx_s]
    k = control_kinematics_func(model, q)
	k_input = u[model.idx_u][9:10]

	e1 = k[1:2] - k_input
	e2 = k[3:4] - k_input
	e3 = k[5:6] - k_input
	e4 = k[7:8] - k_input
	# e5 = k[9:10] - k_input
	# e6 = k[11:12] - k_input
	# e7 = k[13:14] - k_input
	# e8 = k[15:16] - k_input

	d1 = e1' * e1
	d2 = e2' * e2
	d3 = e3' * e3
	d4 = e4' * e4
	# d5 = e5' * e5
	# d6 = e6' * e6
	# d7 = e7' * e7
	# d8 = e8' * e8

	u_ctrl = u[model.idx_u]

	c[1] = s[1] - u_ctrl[1] * d1
	c[2] = s[1] + u_ctrl[1] * d1


	c[3] = s[1] - u_ctrl[2] * d1
	c[4] = s[1] + u_ctrl[2] * d1

	c[5] = s[1] - u_ctrl[3] * d2
	c[6] = s[1] + u_ctrl[3] * d2

	c[7] = s[1] - u_ctrl[4] * d2
	c[8] = s[1] + u_ctrl[4] * d2

	c[9] = s[1] - u_ctrl[5] * d3
	c[10] = s[1] + u_ctrl[5] * d3

	c[11] = s[1] - u_ctrl[6] * d3
	c[12] = s[1] + u_ctrl[6] * d3

	c[13] = s[1] - u_ctrl[7] * d4
	c[14] = s[1] + u_ctrl[7] * d4

	c[15] = s[1] - u_ctrl[8] * d4
	c[16] = s[1] + u_ctrl[8] * d4

    nothing
end

n_ctrl_comp = 16
con_ctrl_comp = stage_constraints(control_comp_con!, n_ctrl_comp, (1:16), t_idx)

function control_limits_con!(c, x, u, t)
	q = x[nq .+ (1:nq)]
	θ = q[3]
	R = rotation_matrix(θ)
	u_ctrl = u[model.idx_u]

	c[1] = -1.0 * (R' * u_ctrl[1:2])[1]
	c[2] = (model.μ[end] * u_ctrl[1])^2.0 - u_ctrl[2]^2.0

	c[3] = -1.0 * (R' * u_ctrl[3:4])[2]
	c[4] = (model.μ[end] * u_ctrl[4])^2.0 - u_ctrl[3]^2.0

	c[5] = (R' * u_ctrl[5:6])[1]
	c[6] = (model.μ[end] * u_ctrl[5])^2.0 - u_ctrl[6]^2.0

	c[7] = (R' * u_ctrl[7:8])[2]
	c[8] = (model.μ[end] * u_ctrl[8])^2.0 - u_ctrl[7]^2.0
end

n_ctrl_lim = 8
con_ctrl_lim = stage_constraints(control_limits_con!, n_ctrl_lim, (1:8), t_idx)

con_contact = contact_constraints(model, T)
con = multiple_constraints([con_contact, con_ctrl_comp, con_ctrl_lim])

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
q = state_to_configuration(x̄)
u = [u[model.idx_u] for u in ū]
γ = [u[model.idx_λ] for u in ū]
b = [u[model.idx_b] for u in ū]
h̄ = h

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
open(vis)
visualize!(vis, model,
    q, u,
    Δt = h)

plot(hcat(q...)', color = :red, width = 1.0, labels = "")
plot(hcat([u..., u[end]]...)[model.idx_u[1:8], :]', linetype = :steppost, color = :black, width = 1.0, labels = "")
