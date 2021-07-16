using Plots

# Model
include_model("planar_push_block_v3")

# Horizon
T = 26

# Time step
tf = 2.5
h = tf / (T-1)


# Bounds
_uu = Inf * ones(model.m)
_uu[model.idx_u] .= Inf
_uu[model.idx_u[1:2]] .= 100.0
_uu[model.idx_λ[1:4]] .= 1.0
_ul = zeros(model.m)
_ul[model.idx_u] .= -Inf
_ul[model.idx_u[1:2]] .= -100.0
_ul[model.idx_λ[1:4]] .= 1.0

ul, uu = control_bounds(model, T, _ul, _uu)

# Initial and final states
q1 = [0.0, 0.0, 0.0, 0.0, -r-1.0e-8]
x1 = [q1; q1]
qT = [0.0, 0.0, 1.0 * π, 0.0, -r-1.0e-8]

# qT = [1.0, 1.0, 0.5 * π, 1.0 - 2.0 * r, 1.0 - 2.0 * r]
xT = [qT; qT]
xl, xu = state_bounds(model, T, x1 = x1, xT = xT)
ϕ_func(model, q1)


# Objective
include_objective("velocity")
obj_velocity = velocity_objective(
    [t > T / 2 ? Diagonal(1.0 * ones(model.nq)) : Diagonal(1.0 * ones(model.nq)) for t = 1:T-1],
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

obj_penalty = PenaltyObjective(1.0e4, model.m)
obj = MultiObjective([obj_tracking, obj_penalty, obj_velocity])

# Constraints
include_constraints(["contact", "stage"])
t_idx = vcat([t for t = 1:T-1])
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
    q, u,
    Δt = h,
	r = model.block_dim[1] + model.block_rnd)

settransform!(vis["/Cameras/default"],
	compose(Translation(0.0, 0.0, 50.0), LinearMap(RotZ(0.5 * pi) * RotY(-pi/2.5))))
setprop!(vis["/Cameras/default/rotated/<object>"], "zoom", 25)

plot(hcat(q̄...)', color = :red, width = 1.0, labels = "")
plot(hcat([ū..., ū[end]]...)[model.idx_u[1:2], :]', linetype = :steppost, color = :black, width = 1.0, labels = "")
