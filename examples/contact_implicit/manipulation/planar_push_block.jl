using Plots

# Model
include_model("planar_push_block")

# Horizon
T = 11

# Time step
tf = 1.0
h = tf / (T-1)


# Bounds
_uu = Inf * ones(model.m)
_uu[model.idx_u] .= Inf
_uu[model.idx_u[1:16]] .= 0.25
_ul = zeros(model.m)
_ul[model.idx_u] .= -Inf
_ul[model.idx_u[1:16]] .= -0.25


ul, uu = control_bounds(model, T, _ul, _uu)

# Initial and final states
q1 = [0.0, 0.0, 0.0]
x1 = [q1; q1]
qT = [1.0; 0.0; 0.0]
xT = [qT; qT]
xl, xu = state_bounds(model, T, x1 = x1, xT = xT)

# Objective
obj_tracking = quadratic_tracking_objective(
    [Diagonal(1.0 * ones(model.n)) for t = 1:T],
    [Diagonal(0.0001 * ones(model.m)) for t = 1:T-1],
    [xT for t = 1:T],
    [zeros(model.m) for t = 1:T])

obj_penalty = PenaltyObjective(1.0e5, model.m)
obj = MultiObjective([obj_tracking, obj_penalty])

# Constraints
include_constraints(["contact", "stage"])
t_idx = vcat([t for t = 1:T-1])

function control_comp_con!(c, x, u, t)
	q = x[nq .+ (1:nq)]
	s = u[model.idx_s]
    k = control_kinematics_func(model, q)
	k_input = u[model.idx_u][17:18]

	e1 = k[1:2] - k_input
	e2 = k[3:4] - k_input
	e3 = k[5:6] - k_input
	e4 = k[7:8] - k_input
	e5 = k[9:10] - k_input
	e6 = k[11:12] - k_input
	e7 = k[13:14] - k_input
	e8 = k[15:16] - k_input

	d1 = e1' * e1
	d2 = e2' * e2
	d3 = e3' * e3
	d4 = e4' * e4
	d5 = e5' * e5
	d6 = e6' * e6
	d7 = e7' * e7
	d8 = e8' * e8

	c[1] = s[1] - u[1] * d1
	c[2] = s[1] - u[2] * d1

	c[3] = s[1] - u[3] * d2
	c[4] = s[1] - u[4] * d2

	c[5] = s[1] - u[5] * d3
	c[6] = s[1] - u[6] * d3

	c[7] = s[1] - u[7] * d4
	c[8] = s[1] - u[8] * d4

	c[9] = s[1] - u[9] * d5
	c[10] = s[1] - u[10] * d5

	c[11] = s[1] - u[11] * d6
	c[12] = s[1] - u[12] * d6

	c[13] = s[1] - u[13] * d7
	c[14] = s[1] - u[14] * d7

	c[15] = s[1] - u[15] * d8
	c[16] = s[1] - u[16] * d8

    nothing
end

n_ctrl_comp = 16
con_ctrl_comp = stage_constraints(control_comp_con!, n_ctrl_comp, (1:16), t_idx)

function control_limits_con!(c, x, u, t)
	q = x[nq .+ (1:nq)]
	θ = q[3]
	R = rotation_matrix(θ)
	u_ctrl = u[model.idx_u]


	c[1:2] = [-1.0; -1.0] .* (R * u_ctrl[1:2])
	c[3:4] = [1.0; -1.0] .* (R * u_ctrl[3:4])
	c[5:6] = [1.0; 1.0] .* (R * u_ctrl[5:6])
	c[7:8] = [-1.0; 1.0] .* (R * u_ctrl[7:8])

	c[9] = -1.0 * (R * u_ctrl[9:10])[1]
	c[10] = -1.0 * (R * u_ctrl[11:12])[2]
	c[11] = (R * u_ctrl[13:14])[1]
	c[12] = (R * u_ctrl[15:16])[2]
end

n_ctrl_lim = 12
con_ctrl_lim = stage_constraints(control_limits_con!, n_ctrl_lim, (1:12), t_idx)

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
@time z̄, info = solve(prob, copy(z0), tol = 1.0e-5, c_tol = 1.0e-5)

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
visualize!(vis, model,
    q̄, ū,
    Δt = h)

settransform!(vis["/Cameras/default"],
	compose(Translation(0.0, 0.0, 50.0), LinearMap(RotZ(0.5 * pi) * RotY(-pi/2.5))))
setprop!(vis["/Cameras/default/rotated/<object>"], "zoom", 25)

plot(hcat(q̄...)', color = :red, width = 1.0, labels = "")
plot(hcat([ū..., ū[end]]...)[model.idx_u[1:16], :]', linetype = :steppost, color = :black, width = 1.0, labels = "")
