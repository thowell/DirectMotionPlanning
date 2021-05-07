# Model
include_model("hopper3D")

function surf(x)
	m = 1.0

	if x[1] >= 0.0
		return m * x[1]
	else
		return -m * x[1]
	end
end

function surf_grad(x)
	m = 1.0

	if x[1] >= 0.0
		return [m; 0.0]
	else
		return [-m; 0.0]
	end
end

model = Hopper3D{Discrete, FixedTime}(n, m, d,
			mb, ml, Jb, Jl,
			μ, g,
			qL, qU,
            nq, nu, nc, nf, nb,
            idx_u,
            idx_λ,
            idx_b,
            idx_ψ,
            idx_η,
            idx_s,
			surf,
			surf_grad)

model_ft = free_time_model(model)

function ϕ_func(model::Hopper3D, q)
	k = kinematics(model, q)
    @SVector [k[3] - model.surf(k[1:2])]
end

# Horizon
T = 31

# Time step
tf = 1.0
h = tf / (T - 1)

# Bounds
_uu = Inf * ones(model_ft.m)
_uu[end] = 2.0 * h
_ul = zeros(model_ft.m)
_ul[model_ft.idx_u] .= -Inf
_ul[end] = 0.5 * h
ul, uu = control_bounds(model_ft, T, _ul, _uu)

# Initial and final states
mrp_init = MRP(UnitQuaternion(RotZ(0.0) * RotY(0.0) * RotX(0.0)))

z_h = 0.0
# q1 = [0.0, 0.0, 0.5, mrp_init.x, mrp_init.y, mrp_init.z, 0.5]
# ϕ_func(model, q1)
# x1 = [q1; q1]
_qT = [0.5, 0.0, 0.0, mrp_init.x, mrp_init.y, mrp_init.z, 0.5]

zh = -1.0 * ϕ_func(model_ft, _qT)[1]

q1 = [-0.5, 0.0, zh, mrp_init.x, mrp_init.y, mrp_init.z, 0.5]
ϕ_func(model_ft, q1)
x1 = [q1; q1]

qT = [0.5, 0.0, zh, mrp_init.x, mrp_init.y, mrp_init.z, 0.5]
ϕ_func(model_ft, qT)
xT = [qT; qT]

xl, xu = state_bounds(model_ft, T,
    [model_ft.qL; model_ft.qL], [model_ft.qU; model_ft.qU])#,
    # x1 = x1, xT = xT)

# Objective
include_objective(["velocity", "nonlinear_stage", "control_velocity"])

qp = [1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0]
obj_tracking = quadratic_time_tracking_objective(
    [Diagonal(0.0 * [qp; qp]) for t = 1:T],
    [Diagonal([1.0e-2, 1.0e-2, 1.0e-2,
		1.0e-3 * ones(model_ft.nc)..., 1.0e-3 * ones(model_ft.nb)...,
		zeros(model_ft.m - model_ft.nu - model_ft.nc - model_ft.nb - 1)..., 0.0])
		for t = 1:T-1],
    [[qT; qT] for t = 1:T],
    [zeros(model_ft.m) for t = 1:T],
    1.0)

obj_penalty = PenaltyObjective(1.0e5, model_ft.m - 1)

obj_velocity = velocity_objective(
    [Diagonal(1.0e-1 * ones(model_ft.nq)) for t = 1:T-1],
    model_ft.nq,
    h = h,
    idx_angle = collect([3]))

obj = MultiObjective([obj_tracking, obj_penalty, obj_penalty])

# Constraints
include_constraints(["free_time", "contact", "stage", "loop"])
con_free_time = free_time_constraints(T)
con_contact = contact_constraints(model_ft, T)

p1_ref = kinematics(model_ft, q1)
pT_ref = kinematics(model_ft, qT)

function pinned1!(c, x, u, t)
    q = view(x, 1:7)
    c[1:3] = p1_ref - kinematics(model_ft, q)
	nothing
end

function pinnedT!(c, x, u, t)
    q = view(x, 7 .+ (1:7))
	c[1:3] = pT_ref - kinematics(model_ft, q)
	nothing
end

T_fix = 5
n_stage = 3
t_idx1 = vcat([t for t = 1:T_fix]...)
t_idxT = vcat([(T - T_fix + 1):T]...)

con_pinned1 = stage_constraints(pinned1!, n_stage, (1:0), t_idx1)
con_pinnedT = stage_constraints(pinnedT!, n_stage, (1:0), t_idxT)

con_loop = loop_constraints(model, collect([(2:7)...,(9:14)...]), 1, T)

con = multiple_constraints([con_free_time,
	con_contact,
	con_pinned1,
	con_pinnedT,
	con_loop])

# Problem
prob = trajectory_optimization_problem(model_ft,
               obj,
               T,
               h = h,
               xl = xl,
               xu = xu,
               ul = ul,
               uu = uu,
               con = con)

# Trajectory initialization
q_ref = linear_interpolation(q1, qT, T+1)
x0 = configuration_to_state(q_ref) # linear interpolation on state
u0 = [[1.0e-3 * rand(model.m); h] for t = 1:T-1] # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem

@time z̄, info = solve(prob, copy(z0),
	tol = 1.0e-3, c_tol = 1.0e-3, mapl = 5)

check_slack(z̄, prob)
x̄, ū = unpack(z̄, prob)
tf, t, h̄ = get_time(ū)

q = state_to_configuration(x̄)
u = [u[model.idx_u] for u in ū]
γ = [u[model.idx_λ] for u in ū]
b = [u[model.idx_b] for u in ū]
ψ = [u[model.idx_ψ] for u in ū]
η = [u[model.idx_η] for u in ū]
h̄ = mean(h̄)

# include(joinpath(pwd(), "models/visualize.jl"))
# vis = Visualizer()
# render(vis)
visualize!(vis, model_ft,
	state_to_configuration(x̄),
	q,
	Δt = h̄)

# f(x) = x[3] - model_ft.surf(x[1:2])[1]
#
# xlims = [-1.0, 1.0]
# ylims = [-2.0, 2.0]
# mesh = GeometryBasics.Mesh(f,
# 	HyperRectangle(Vec(xlims[1], ylims[1], -2.0), Vec(xlims[2]-xlims[1], ylims[2]-ylims[1], 4.0)),
#     Meshing.MarchingCubes(), samples=(200, 200, Int(floor(200/8))))
# setobject!(vis["surface"], mesh,
# 	MeshPhongMaterial(color=RGBA{Float32}(1.0, 0.0, 0.0, 0.5)))

ref = Diagonal([-1.0; 1.0; 1.0])
mrp_init = MRP(RotZ(0.0) * RotY(-0.25 * π) * RotX(-0.0 * π))
# mrp_mirror = MRP(mrp_init')
mrp_mirror = MRP(mrp_init * ref)
q1 = [0.0, 0.0, 1.0, mrp_init.x, mrp_init.y, mrp_init.z, 0.5]
qT = [0.0, 0.0, 1.0, mrp_mirror.x, mrp_mirror.y, mrp_mirror.z, 0.5]

visualize!(vis, model_ft,
	[q1],
	# qa,
	# q_ref,
	Δt = h̄)


qa = linear_interpolation(q1, qT, 10)
