using Plots

# Model
include_model("bimanipulation_block")

# add obstacle
function ϕ_func(model::BimanipulationBlock, q)
    p = view(q, 1:2)
	θ = q[3]
    R = rotation_matrix(θ)

	xl = 0.25
	xu = 0.75
	h = 0.05

	k1 = p + R * model.contact_corner_offset[1]
	k2 = p + R * model.contact_corner_offset[2]
	k3 = p + R * model.contact_corner_offset[3]
	k4 = p + R * model.contact_corner_offset[4]

	SVector{4}([k1[2] - ((k1[1] >= xl && k1[1] <= xu) ? h : 0.0),
			    k2[2] - ((k2[1] >= xl && k2[1] <= xu) ? h : 0.0),
			    k3[2] - ((k3[1] >= xl && k3[1] <= xu) ? h : 0.0),
			    k4[2] - ((k4[1] >= xl && k4[1] <= xu) ? h : 0.0)])
end

# Horizon
T = 11

# Time step
tf = 1.0
h = tf / (T-1)

# Bounds
_uu = Inf * ones(model.m)
_uu[model.idx_u] .= Inf
# _uu[model.idx_u[1:8]] .= 1000.0
# _uu[model.idx_u[3:4]] .= 0.0
# _uu[model.idx_u[7:8]] .= 0.0
# _uu[model.idx_λ] .= 1.0
_ul = zeros(model.m)
_ul[model.idx_u] .= -Inf
# _ul[model.idx_u[1:8]] .= -1000.0
# _ul[model.idx_u[3:4]] .= 0.0
# _ul[model.idx_u[7:8]] .= 0.0
# _ul[model.idx_λ] .= 1.0

_ul[model.idx_u] .= 0.0
_uu[model.idx_u] .= 0.0

_ul[model.idx_u[9:10]] = x0[t][nq .+ (1:2)] + model.control_input_offset[1]
_ul[model.idx_u[11:12]] = x0[t][nq .+ (1:2)] + model.control_input_offset[3]
_ul[model.idx_u[1:2]] = [-10.0; 0.5 * model.mass * model.g * h]
_ul[model.idx_u[5:6]] = [10.0; 0.5 * model.mass * model.g * h]

_uu[model.idx_u[9:10]] = x0[t][nq .+ (1:2)] + model.control_input_offset[1]
_uu[model.idx_u[11:12]] = x0[t][nq .+ (1:2)] + model.control_input_offset[3]
_uu[model.idx_u[1:2]] = [-10.0; 0.5 * model.mass * model.g * h]
_uu[model.idx_u[5:6]] = [10.0; 0.5 * model.mass * model.g * h]

ul, uu = control_bounds(model, T, _ul, _uu)

# Initial and final states
q1 = [0.0, 2.0 * r, 0.0]
x1 = [q1; q1]
qT = [0.0; 2.0 * r; 0.0]
xT = [qT; qT]
qM = [0.0, 2.0 * r, 0.0]

# Trajectory initialization
q0 = [q1, linear_interpolation(q1, qM, 13)..., linear_interpolation(qM, qT, 12)..., qT]
q0 = [q1 for t = 1:T+1]
x0 = configuration_to_state(q0)

u0 = [0.0 * rand(model.m) for t = 1:T-1] # random controls
for t = 1:T-1
	u0[t][model.idx_u] .= 0.0
	u0[t][model.idx_u[9:10]] = x0[t][nq .+ (1:2)] + model.control_input_offset[1]
	u0[t][model.idx_u[11:12]] = x0[t][nq .+ (1:2)] + model.control_input_offset[3]
	u0[t][model.idx_u[1:2]] = [-1.0; 0.5 * model.mass * model.g * h]
	u0[t][model.idx_u[5:6]] = [1.0; 0.5 * model.mass * model.g * h]
end

xl, xu = state_bounds(model, T, x1 = x1, xT = xT)

# qm1 = zeros(nq)
# vm1 = zeros(nq)
# qm2 = zeros(nq)
# vm2 = zeros(nq)
#
# D1L1, D2L1 = lagrangian_derivatives(model, qm1, vm1)
# D1L2, D2L2 = lagrangian_derivatives(model, qm2, vm2)
# 0.5 * h * D1L1 + D2L1 + 0.5 * h * D1L2 - D2L2
# B_func(model, q1)' * u0[1][model.idx_u[1:8]]
# k = control_kinematics_func(model, q1)

# Objective
include_objective(["velocity", "control_velocity", "nonlinear_stage"])

function l_stage(x, u, t)
	q2 = view(x, nq .+ (1:nq))
	k = control_kinematics_func(model, q2)
	k1 = k[1:2]
	p1 = u[model.idx_u[9:10]]

	k3 = k[5:6]
	p2 = u[model.idx_u[11:12]]

    J = 0.0
	u_ctrl_err = u[model.idx_u] - u0[t][model.idx_u]
	J += 1.0 * u_ctrl_err' * u_ctrl_err
	# J += 1000.0 * (k1 - p1)' * (k1 - p1)
	# J += 1000.0 * (k3 - p2)' * (k3 - p2)

    return J
end

l_terminal(x) = 0.0
obj_stage = nonlinear_stage_objective(l_stage, l_terminal)

# obj_velocity = velocity_objective(
#     [t > T / 2 ? Diagonal(1.0 * ones(model.nq)) : Diagonal(1.0 * ones(model.nq)) for t = 1:T-1],
#     model.nq,
#     h = h,
#     idx_angle = collect([3]))
#
obj_tracking = quadratic_tracking_objective(
    [Diagonal(1.0 * ones(model.n)) for t = 1:T],
    # [Diagonal(0.1 * ones(model.m)) for t = 1:T-1],
	[Diagonal([0.0 * ones(model.nu);
		zeros(model.nc);
		zeros(model.nb);
		zeros(model.m - model.nu - model.nc - model.nb)]) for t = 1:T-1],
    [x0[t] for t = 1:T],
    [zeros(model.m) for t = 1:T])

# obj_ctrl_vel = control_velocity_objective(Diagonal([1.0 * ones(model.nu); 0.0 * ones(model.nc + model.nb); zeros(model.m - model.nu - model.nc - model.nb)]))

obj_penalty = PenaltyObjective(1.0e5, model.m)
obj = MultiObjective([obj_penalty])#, obj_stage, obj_tracking])

# Constraints
include_constraints(["contact", "stage"])
t_idx = vcat([t for t = 1:T-1])

function control_comp_con!(c, x, u, t)
	q = x[nq .+ (1:nq)]
	s = u[model.idx_s]
    k = control_kinematics_func(model, q)
	k_input1 = u[model.idx_u][9:10]
	k_input2 = u[model.idx_u][11:12]

	u_ctrl = u[model.idx_u]

	e1 = k[1:2] - k_input1
	e2 = k[3:4] - k_input1
	e3 = k[5:6] - k_input1
	e4 = k[7:8] - k_input1

	d1 = e1' * e1
	d2 = e2' * e2
	d3 = e3' * e3
	d4 = e4' * e4

	# input 1
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

	# input 2
	f1 = k[1:2] - k_input2
	f2 = k[3:4] - k_input2
	f3 = k[5:6] - k_input2
	f4 = k[7:8] - k_input2

	g1 = f1' * f1
	g2 = f2' * f2
	g3 = f3' * f3
	g4 = f4' * f4


	c[17] = s[1] - u_ctrl[1] * g1
	c[18] = s[1] + u_ctrl[1] * g1

	c[19] = s[1] - u_ctrl[2] * g1
	c[20] = s[1] + u_ctrl[2] * g1

	c[21] = s[1] - u_ctrl[3] * g2
	c[22] = s[1] + u_ctrl[3] * g2

	c[23] = s[1] - u_ctrl[4] * g2
	c[24] = s[1] + u_ctrl[4] * g2

	c[25] = s[1] - u_ctrl[5] * g3
	c[26] = s[1] + u_ctrl[5] * g3

	c[27] = s[1] - u_ctrl[6] * g3
	c[28] = s[1] + u_ctrl[6] * g3

	c[29] = s[1] - u_ctrl[7] * g4
	c[30] = s[1] + u_ctrl[7] * g4

	c[31] = s[1] - u_ctrl[8] * g4
	c[32] = s[1] + u_ctrl[8] * g4

    nothing
end

n_ctrl_comp = 32
con_ctrl_comp = stage_constraints(control_comp_con!, n_ctrl_comp, (1:32), t_idx)

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
con = multiple_constraints([con_contact])#, con_ctrl_comp, con_ctrl_lim])

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
# open(vis)

visualize!(vis, model,
    # q0, u0,
	q, u,
	u_mag = 1.0,
    Δt = h)

# setobject!(vis["obstacle"],
# 	GeometryBasics.HyperRectangle(Vec(0.0, 0.0, 0.0), Vec(0.5, 0.5, 0.05)),
# 	MeshPhongMaterial(color = RGBA(0.7, 0.7, 0.7, 1.0)))
# settransform!(vis["obstacle"], Translation(0.25, -0.25, 0))

plot(hcat(q...)', color = :red, width = 1.0, labels = "")
plot(hcat([u..., u[end]]...)[model.idx_u[1:8], :]', linetype = :steppost, color = :black, width = 1.0, labels = "")
