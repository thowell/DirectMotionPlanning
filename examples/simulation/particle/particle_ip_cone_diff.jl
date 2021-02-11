using Plots
include("model.jl")
include("second_order_cone.jl")
include("step.jl")
include("simulate.jl")
include("visualize.jl")

# model
model = Particle(2 * 3, 3, 0, 1.0, 9.81, 0.5, 3, 3)
joinpath(@__DIR__,"model.jl")

# horizon
T = 11

# time step
h = 0.1

# initial conditions
v1 = [0.0; 0.0; 0.0]
q1 = [0.0; 0.0; 1.0]

v2 = v1 - gravity(model, q1) * h
q2 = q1 + 0.5 * (v1 + v2) * h

# # simulate
q_sol, y_sol, b_sol, Δq1, Δq2, Δu1 = simulate(q1, q2, T, h)

# include(joinpath(pwd(), "models/visualize.jl"))
# vis = Visualizer()
# render(vis)
#
# visualize!(vis, model,
#     q_sol,
#     Δt = h)
#
# settransform!(vis["/Cameras/default"],
#     compose(Translation(0.0, 0.0, -1.0), LinearMap(RotZ(pi))))
#
# settransform!(vis["/Cameras/default"],
# 	compose(Translation(0.0, 0.0, 3.0),LinearMap(RotY(-pi/2.5))))

## trajectory optimization
ul, uu = control_bounds(model, T, [-Inf; -Inf; 0.0], [Inf; Inf; 0.0])
x1 = [q1; q2]
xl, xu = state_bounds(model, T, x1 = x1)

# Objective
obj = quadratic_tracking_objective(
        [t < T ? Diagonal(1.0e-1 * ones(model.n)) : 10.0 * Diagonal(ones(model.n)) for t = 1:T],
        [Diagonal(1.0e-3 * ones(model.m)) for t = 1:T-1],
        [[1.0; 1.0; 0.0; 1.0; 1.0; 0.0] for t = 1:T],
        [zeros(model.m) for t = 1:T])

# Constraints
include("simulator_dynamics.jl")
con_sim = dynamics_constraints(model, T)

# Problem
prob = trajectory_optimization_problem(model,
               obj,
               T,
               h = h,
               xl = xl,
               xu = xu,
               ul = ul,
               uu = uu,
               con = con_sim,
			   dynamics = false)

# Trajectory initialization
x0 = [0.01 * rand(model.n) for t = 1:T] #linear_interpolation(x1, x1, T) # linear interpolation on state
u0 = [1.0 * rand(model.m) for t = 1:T-1] # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem
@time z̄, info = solve(prob, copy(z0), tol = 1.0e-3, c_tol = 1.0e-3)

x̄, ū = unpack(z̄, prob)
q̄ = state_to_configuration(x̄)

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)

visualize!(vis, model,
    q̄,
    Δt = h)

@show q̄[end]
