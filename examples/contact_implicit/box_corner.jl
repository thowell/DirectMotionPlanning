# Model
include_model("box")

# Horizon
T = 26

# Time step
tf = 2.5
h = tf / (T-1)

# ul <= u <= uu
_uu = Inf * ones(model.m)
_ul = zeros(model.m)
_ul[model.idx_u] .= -Inf
ul, uu = control_bounds(model, T, _ul, _uu)

# Initial and final states
mrp_init = MRP(UnitQuaternion(RotY(0.0) * RotX(0.0)))
mrp_side = MRP(UnitQuaternion(RotY(-1.0 * atan(1.0 / sqrt(2.0))) * RotX(pi / 4.0)))

mrp_corner = MRP(UnitQuaternion(RotY(-1.0 * atan(1.0 / sqrt(2.0))) * RotX(pi / 4.0)))

q1 = [model.r, model.r, model.r, mrp_init.x, mrp_init.y, mrp_init.z]
qT = [0.0, 0.0, model.r * sqrt(3.0), mrp_corner.x, mrp_corner.y, mrp_corner.z]

x1 = [q1; q1]

xl, xu = state_bounds(model, T, x1 = x1)

# Objective
obj_penalty = PenaltyObjective(100.0, model.m)

Qq = Diagonal(ones(model.nq))
Q = cat(0.5 * Qq, 0.5 * Qq, dims = (1, 2))
QT = cat(0.5 * Qq, 100.0 * Diagonal(ones(model.nq)), dims = (1, 2))
R = Diagonal([1.0e-1 * ones(model.nu)..., zeros(model.m - model.nu)...])

obj_tracking = quadratic_tracking_objective(
    [t < T ? Q : QT for t = 1:T],
    [R for t = 1:T-1],
    [[zeros(model.nq); qT] for t = 1:T],
    [zeros(model.m) for t = 1:T])

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
x0 = linear_interpolation(x1, [qT; qT], T) # linear interpolation on state
u0 = [0.001 * rand(model.m) for t = 1:T-1] # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem

@time z̄ , info = solve(prob, copy(z0),
    tol = 1.0e-3, c_tol = 1.0e-3, mapl = 5)

check_slack(z̄, prob)

x̄, ū = unpack(z̄, prob)

# Visualize
include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
visualize!(vis, model,
    state_to_configuration([[x̄[1] for i = 1:10]...,x̄..., [x̄[end] for i = 1:10]...]),
    Δt = h)
open(vis)
