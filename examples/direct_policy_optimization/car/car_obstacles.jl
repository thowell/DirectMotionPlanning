# Model
include_model("car")

# Horizon
T = 51

# Time step
tf = 1.0
h = tf / (T - 1)

# Bounds

# ul <= u <= uu
ul, uu = control_bounds(model, T, -3.0, 3.0)

# Initial and final states
x1 = [0.0; 0.0; 0.0]
xT = [1.0; 1.0; 0.0]
xl, xu = state_bounds(model, T, x1 = x1, xT = xT)

# Objective
obj = quadratic_tracking_objective(
        [(t < T ? Diagonal(ones(model.n))
            : Diagonal(10.0 * ones(model.n))) for t = 1:T],
        [Diagonal(1.0e-1 * ones(model.m)) for t = 1:T-1],
        [xT for t = 1:T], [zeros(model.m) for t = 1:T])

# Constraints
include_constraints("stage")

circles = [(0.85, 0.3, 0.1),
           (0.375, 0.75, 0.1),
           (0.25, 0.2, 0.1),
           (0.75, 0.8, 0.1)]

function circle_obs(x, y, xc, yc, r)
    (x - xc)^2.0 + (y - yc)^2.0 - r^2.0
end

function obstacles!(c, x, u)
    c[1] = circle_obs(x[1], x[2], circles[1][1], circles[1][2], circles[1][3])
    c[2] = circle_obs(x[1], x[2], circles[2][1], circles[2][2], circles[2][3])
    c[3] = circle_obs(x[1], x[2], circles[3][1], circles[3][2], circles[3][3])
    c[4] = circle_obs(x[1], x[2], circles[4][1], circles[4][2], circles[4][3])
    nothing
end

n_stage = 4
n_con = n_stage * (T - 1)
t_idx = [t for t = 1:T-1]
con_obstacles = stage_constraints(obstacles!, n_stage, (1:n_stage), t_idx)

# Problem
prob = trajectory_optimization_problem(
           model,
           obj,
           T,
           h = h,
           xl = xl,
           xu = xu,
           ul = ul,
           uu = uu,
           con = con_obstacles)

# Trajectory initialization
x0 = linear_interpolation(x1, xT, T) # linear interpolation on state
u0 = [0.001 * ones(model.m) for t = 1:T-1]

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

# Solve nominal problem
if false
    @time z̄, info = solve(prob, copy(z0))
    @save joinpath(@__DIR__, "sol_to.jld2") z̄
else
    println("Loading solution...")
    @load joinpath(@__DIR__, "sol_to.jld2") z̄
end
