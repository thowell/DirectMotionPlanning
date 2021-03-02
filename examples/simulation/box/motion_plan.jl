using Plots
include("model.jl")
include("second_order_cone.jl")
include("step.jl")
include("simulate.jl")
include("visualize.jl")

# horizon
T = 201

# time step
h = 0.01
t = range(0, stop = h * (T - 1), length = T)

mrp = MRP(UnitQuaternion(RotY(π / 6.0) * RotX(π / 10.0)))

# initial conditions
v1 = [2.5; 5.0; 0.0; 0.0; 0.0; 0.0]
q1 = [0.0; 0.0; 1.0; mrp.x; mrp.y; mrp.z]

v2 = v1 - gravity(model) * h
q2 = q1 + 0.5 * (v1 + v2) * h

# simulate
q_sol, y_sol, b_sol, Δq1, Δq2, Δu1 = simulate(q1, q2, T, h)

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
T = 11
h = 0.1
ul, uu = control_bounds(model, T)

mrp_init = MRP(UnitQuaternion(RotY(0.0) * RotX(0.0)))
# mrp_init = MRP(UnitQuaternion(RotY(pi / 4.0)))

mrp_corner = MRP(UnitQuaternion(RotY(-1.0 * atan(1.0 / sqrt(2.0))) * RotX(pi / 4.0)))
q1 = [model.r, model.r, model.r, mrp_init.x, mrp_init.y, mrp_init.z]
# qT = [0.0, 0.0, model.r * sqrt(3.0), mrp_corner.x, mrp_corner.y, mrp_corner.z]
qT = copy(q1)
qT[1] = 1.0
qT[2] = 1.0
visualize!(vis, model,
    [q1],
    Δt = h)
x1 = [q1; q1]
xl, xu = state_bounds(model, T, x1 = x1)

# Objective
obj = quadratic_tracking_objective(
        [t < T ? Diagonal(1.0e-1 * ones(model.n)) : 100.0 * Diagonal(ones(model.n)) for t = 1:T],
        [Diagonal(1.0e-2 * ones(model.m)) for t = 1:T-1],
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
# u0 = [1.0 * [0.0; 0.0; 0.0; t == 13 ? 0.0 : 0.0; t == 1 ? 0.0 : 0.0; 0.0] for t = 1:T-1] # random controls
u0 = [[0.1 * rand(3); 0.0; 0.0; 0.0] for t = 1:T-1]
# step(q1, q1, u0[1], h;
#     tol = 1.0e-8, max_iter = 100, z_init = 10.0)
q_sol, y_sol, b_sol, Δq1, Δq2, Δu1 = simulate(q1, q1, T-1, h, u1 = u0, r_tol = 1.0e-5, μ_tol = 1.0e-3, z_init = 1.0e-2)

visualize!(vis, model,
    q_sol,
    Δt = h)

# Pack trajectories into vector
z0 = pack(x0, u0, prob)
prob.num_var
prob.num_con
#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem
include_snopt()
@time z̄, info = solve(prob, copy(z0),
	nlp = :ipopt,
	tol = 1.0e-2, c_tol = 1.0e-2)

cc = zeros(con_sim.n)
constraints!(cc, z̄, con_sim, model, prob.prob.idx, h, T)
norm(cc, Inf)
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
