# Model
include_model("hopper3D")
model = free_time_model(model)

# Horizon
T = 51

# Time step
tf = 1.0
h = tf / (T - 1)

# Bounds
_uu = Inf * ones(model.m)
_ul = zeros(model.m)
_ul[model.idx_u] .= -Inf
_uu[end] = 2.0 * h
_ul[end] = 0.5 * h
ul, uu = control_bounds(model, T, _ul, _uu)

# Initial and final states
mrp_init = MRP(UnitQuaternion(RotZ(0.0) * RotY(0.0) * RotX(0.0)))

z_h = 0.25
q1 = [0.0, 0.0, 0.5 + z_h, mrp_init.x, mrp_init.y, mrp_init.z, 0.25]
x1 = [q1; q1]
# qT = [0.0, 0.0, 0.5 + z_h, mrp_init.x, mrp_init.y, mrp_init.z, 0.5]
# xT = [qT; qT]

xl, xu = state_bounds(model, T,
    [model.qL; model.qL], [model.qU; model.qU],
    x1 = [q1; Inf * ones(nq)])

# Objective
Qq = Diagonal([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
Q = cat(0.5 * Qq, 0.5 * Qq, dims = (1, 2))
QT = cat(0.5 * Qq, 1.0 * Diagonal(ones(model.nq)), dims = (1, 2))
R = Diagonal([1.0, 1.0, 1.0, zeros(model.m - model.nu - 1)..., 0.0])

obj_tracking = quadratic_time_tracking_objective(
    [t < T ? 0.0 * Q : 0.0 * QT for t = 1:T],
    [R for t = 1:T-1],
    [[q1; q1] for t = 1:T],
    [zeros(model.m) for t = 1:T],
    1.0)

obj_velocity = velocity_objective(
    [Diagonal(1.0 * ones(model_ft.nq)) for t = 1:T-1],
    model_ft.nq,
    h = h,
    idx_angle = collect(4:6))

obj_penalty = PenaltyObjective(1.0e5, model.m-1)
obj = MultiObjective([obj_tracking, obj_penalty, obj_velocity])

# Constraints
include_constraints(["contact", "free_time", "loop"])
con_contact = contact_constraints(model_ft, T)
con_free_time = free_time_constraints(T)
con_loop = loop_constraints(model_ft, (1:model_ft.n), 1, T)
con = multiple_constraints([con_contact, con_free_time, con_loop])

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
x0 = [[q1; q1] for t = 1:T] #linear_interpolation(x1, x1, T) # linear interpolation on state
u0 = [1.0e-3 * rand(model.m) for t = 1:T-1] # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem

@time z̄, info = solve(prob, copy(z0), tol = 1.0e-6, c_tol = 1.0e-6, mapl = 5)

check_slack(z̄, prob)
x̄, ū = unpack(z̄, prob)
tf, t, h̄ = get_time(ū)

q = state_to_configuration(x̄)
u = [u[model.idx_u] for u in ū]
γ = [u[model.idx_λ] for u in ū]
b = [u[model.idx_b] for u in ū]
h̄ = mean(h̄)
@save joinpath(pwd(), "examples/trajectories/hopper3D_vertical_gait.jld2") z̄ x̄ ū h̄ q u γ b

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
visualize!(vis, model, state_to_configuration(x̄), Δt = ū[1][end])
