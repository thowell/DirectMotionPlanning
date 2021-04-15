# Model
function fd(model::DoubleIntegrator{Discrete, FixedTime}, x⁺, x, u, w, h, t)
    x⁺ - [x[1] + x[2] + w[1]; x[2] + u[1] + w[2]]
end

function fd(model::DoubleIntegrator{Discrete, FixedTime}, x⁺, x, u, w, h, t)
    x⁺ - [x[1] + x[2] + w[1]; x[2] - u[1] * (x[1]^2.0)  + w[2]]
end

model = DoubleIntegrator{Discrete, FixedTime}(n, m, d)

# Horizon
T = 11

tf = 10.0
h0 = tf / (T-1) # timestep
t = range(0.0, stop = tf, length = T)

# Initial and final states
x1 = [1.0; 0.0]
xT = [0.0; 0.0]

xl, xu = state_bounds(model, T, x1 = x1, xT = xT)

# Objective
obj = quadratic_tracking_objective(
        [t < T ? Diagonal(ones(model.n)) : Diagonal(ones(model.n)) for t = 1:T],
        [Diagonal(1.0 * ones(model.m)) for t = 1:T-1],
        [xT for t = 1:T],
		[zeros(model.m) for t = 1:T])

# Problem
prob = trajectory_optimization_problem(model,
			   obj,
			   T,
               xl = xl,
               xu = xu)

# Initialization
x0 = linear_interpolation(x1, xT, T) # linear interpolation for states
u0 = [ones(model.m) for t = 1:T-1]

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

# Solve
@time z, info = solve(prob, copy(z0))

# Visualize
using Plots
x, u = unpack(z, prob)
plot(t, hcat(x...)', width = 2.0,
	xlabel = "time (s)", ylabel = "state", label = ["x1" "x2"])
plot(t, hcat(u..., u[end])', width = 2.0,
	xlabel = "time (s)", ylabel = "control", label = "u1",
	linetype = :steppost)
