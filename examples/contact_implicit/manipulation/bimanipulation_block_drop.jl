using Plots

# Model
include_model("bimanipulation_block")

# Horizon
T = 26

# Time step
tf = 2.5
h = tf / (T-1)

# Bounds
_uu = Inf * ones(model.m)
_uu[model.idx_u] .= 0.0
_ul = zeros(model.m)
_ul[model.idx_u] .= 0.0
ul, uu = control_bounds(model, T, _ul, _uu)

# Initial and final states
q1 = [0.0, 3 * r, 0.0]
x1 = [q1; q1]
qT = [0.0; r; 0.0]
xT = [qT; qT]

# Trajectory initialization
q0 = [q1, linear_interpolation(q1, qT, T)...]
x0 = configuration_to_state(q0)

u0 = [0.001 * rand(model.m) for t = 1:T-1] # random controls

xl, xu = state_bounds(model, T, x1 = x1, xT = xT)

# Objective
obj_penalty = PenaltyObjective(1.0e4, model.m)
obj = MultiObjective([obj_penalty])

# Constraints
include_constraints(["contact"])
con_contact = contact_constraints(model, T)
con = multiple_constraints([con_contact])

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
	u_mag = 5.0,
    Δt = h)

plot(hcat(q...)', color = :red, width = 1.0, labels = "")
plot(hcat([u..., u[end]]...)[model.idx_u[1:12], :]', linetype = :steppost, color = :black, width = 1.0, labels = "")
