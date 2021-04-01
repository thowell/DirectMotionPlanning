# Model
include_model("particle")

# Horizon
T = 51

# Time step
tf = 5.0
h = tf / (T-1)

# Bounds
_uu = Inf * ones(model.m)
_uu[model.idx_u] .= 0.0
_ul = zeros(model.m)
ul, uu = control_bounds(model, T, _ul, _uu)

# Initial and final states
q1 = [0.0, 0.0, 1.0]
v1 = [3.0, 5.0, 0.0]
v2 = v1 - C_func(model, q1, v1) * h
q2 = q1 + 0.5 * h * (v1 + v2)

x1 = [q1; q2]

xl, xu = state_bounds(model, T, x1 = x1)

# Objective
obj = PenaltyObjective(1.0e5, model.m)

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
x0 = [0.01 * rand(model.n) for t = 1:T] #linear_interpolation(x1, x1, T) # linear interpolation on state
u0 = [0.001 * rand(model.m) for t = 1:T-1] # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem
@time z̄ , info = solve(prob, copy(z0), tol = 1.0e-5, c_tol = 1.0e-5)

check_slack(z̄, prob)
x̄, ū = unpack(z̄, prob)

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
visualize!(vis, model,
    state_to_configuration(x̄),
    Δt = h)

open(vis)
