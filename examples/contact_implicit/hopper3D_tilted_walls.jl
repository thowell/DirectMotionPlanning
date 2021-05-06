# Model
include_model("hopper3D")

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
			x -> 0.1 * x[1:2]' * x[1:2],
			x -> 0.2 * x[1:2])

function ϕ_func(model::Hopper3D, q)
	k = kinematics(model, q)
    @SVector [k[3] - model.surf(k[1:2])]
end

# Horizon
T = 31

# Time step
tf = 3.0
h = tf / (T - 1)

# Bounds
_uu = Inf * ones(model.m)
_ul = zeros(model.m)
_ul[model.idx_u] .= -Inf
ul, uu = control_bounds(model, T, _ul, _uu)

# Initial and final states
mrp_init = MRP(UnitQuaternion(RotZ(0.0) * RotY(0.0) * RotX(0.0)))

z_h = 0.0
q1 = [0.0, 0.0, 0.5, mrp_init.x, mrp_init.y, mrp_init.z, 0.5]
ϕ_func(model, q1)
x1 = [q1; q1]
_qT = [1.0, 1.0, 0.0, mrp_init.x, mrp_init.y, mrp_init.z, 0.5]
zh = -1.0 * ϕ_func(model, _qT)[1]
qT = [1.0, 1.0, zh, mrp_init.x, mrp_init.y, mrp_init.z, 0.5]
ϕ_func(model, qT)
xT = [qT; qT]

xl, xu = state_bounds(model, T,
    [model.qL; model.qL], [model.qU; model.qU],
    x1 = x1, xT = xT)

# Objective
Qq = Diagonal([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
Q = cat(0.5 * Qq, 0.5 * Qq, dims = (1, 2))
QT = cat(0.5 * Qq, 100.0 * Diagonal(ones(model.nq)), dims = (1, 2))
R = Diagonal([1.0e-1, 1.0e-1, 1.0e-3, zeros(model.m - model.nu)...])

obj_tracking = quadratic_tracking_objective(
    [t < T ? Q : QT for t = 1:T],
    [R for t = 1:T-1],
    [xT for t = 1:T],
    [zeros(model.m) for t = 1:T])

obj_penalty = PenaltyObjective(1.0e5, model.m)

obj = MultiObjective([obj_tracking, obj_penalty])

# Constraints
include_constraints("contact")
con_contact = contact_constraints(model, T)

# Problem
prob = trajectory_optimization_problem(model,
               obj,
               T,
               h = h,
               xl = xl,
               xu = xu,
               ul = ul,
               uu = uu,
               con = con_contact)

# Trajectory initialization
q_ref = linear_interpolation(q1, qT, T+1)
x0 = configuration_to_state(q_ref) # linear interpolation on state
u0 = [1.0e-3 * rand(model.m) for t = 1:T-1] # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem

@time z̄, info = solve(prob, copy(z0), tol = 1.0e-3, c_tol = 1.0e-3, mapl = 5)

check_slack(z̄, prob)
x̄, ū = unpack(z̄, prob)
q = state_to_configuration(x̄)

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
visualize!(vis, model, state_to_configuration(x̄), Δt = h)

f(x) = x[3] - model.surf(x[1:2])[1]

xlims = [-2.0, 2.0]
ylims = [-2.0, 2.0]
mesh = GeometryBasics.Mesh(f,
	HyperRectangle(Vec(xlims[1], ylims[1], -2.0), Vec(xlims[2]-xlims[1], ylims[2]-ylims[1], 4.0)),
    Meshing.MarchingCubes(), samples=(200, 200, Int(floor(200/8))))
setobject!(vis["surface"], mesh,
	MeshPhongMaterial(color=RGBA{Float32}(1.0, 0.0, 0.0, 0.5)))
