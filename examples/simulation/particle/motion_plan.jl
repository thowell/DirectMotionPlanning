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
T = 101

# time step
h = 0.01
t = range(0, stop = h * (T - 1), length = T)

# initial conditions
v1 = [0.0; 0.0; 0.0]
q1 = [0.0; 0.0; 1.0]

v2 = v1 - gravity(model, q1) * h
q2 = q1 + 0.5 * (v1 + v2) * h

# # simulate
q_sol, y_sol, b_sol, Δq1, Δq2, Δu1 = simulate(q1, q2, T, h)

plot(hcat(q_sol...)[:, 1:1:end]')
include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)

visualize!(vis, model,
    q_sol,
    Δt = h)

settransform!(vis["/Cameras/default"],
    compose(Translation(0.0, 0.0, -1.0), LinearMap(RotZ(pi))))

settransform!(vis["/Cameras/default"],
	compose(Translation(0.0, 0.0, 3.0),LinearMap(RotY(-pi/2.5))))

## trajectory optimization
T = 26
h = 0.1
ul, uu = control_bounds(model, T, zeros(model.m), zeros(model.m))
ul[1] = [-Inf; -Inf; 0.0]
uu[1] = [Inf; Inf; 0.0]
x1 = [q1; q2]
xl, xu = state_bounds(model, T, x1 = x1)
qT = [1.0; 2.0; 0.0]

# Objective
obj = quadratic_tracking_objective(
        [t < T ? Diagonal(1.0e-2 * ones(model.n)) : 100.0 * Diagonal(ones(model.n)) for t = 1:T],
        [Diagonal(1.0e-3 * ones(model.m)) for t = 1:T-1],
        [[qT; qT] for t = 1:T],
        [zeros(model.m) for t = 1:T])

# Constraints
include("simulator_dynamics.jl")
con_sim = simulator_constraints(model, T)

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
x0 = configuration_to_state([q1, linear_interpolation(q1, qT, T)...]) #linear_interpolation(x1, x1, T) # linear interpolation on state
u0 = [1.0e-1 * rand(model.m) for t = 1:T-1] # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem
include_snopt()
@time z̄, info = solve(prob, copy(z0),
	nlp = :SNOPT7,
	tol = 1.0e-3, c_tol = 1.0e-3)

x̄, ū = unpack(z̄, prob)
q̄ = state_to_configuration(x̄)
t = range(0, stop = h * (T - 1), length = T)

plot(t, hcat(q̄[2:end]...)', xlabel = "time (s)", ylabel = "")
plot(t, hcat(ū...,ū[end])',
	xlabel = "time (s)" , ylabel = "", linetype = :steppost)

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)

visualize!(vis, model,
    [[q̄[1] for i = 1:10] ..., q̄...],
    Δt = h)

@show q̄[end]

settransform!(vis["/Cameras/default"],
    compose(Translation(0.0, 0.0, -1.0), LinearMap(RotZ(-pi))))

settransform!(vis["/Cameras/default"],
	compose(Translation(0.0, 0.0, 1.0),LinearMap(RotY(-pi/2.5))))

setobject!(vis["particle_goal"],
	Rect(Vec(0, 0, 0),Vec(0.2, 0.2, 0.2)),
	MeshPhongMaterial(color = RGBA(0.0, 1.0, 1.0, 0.5)))

settransform!(vis["particle_goal"], Translation(qT...))

open(vis)
