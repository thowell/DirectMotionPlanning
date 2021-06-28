# Model
include_model("double_integrator")

# Horizon
T = 11

tf = 1.0
h0 = tf / (T-1) # timestep

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
include_snopt() # set path
@time z, info = solve(prob, copy(z0), nlp = :SNOPT7)

# Visualize
using Plots
x, u = unpack(z, prob)
plot(hcat(x...)', width = 2.0)
plot(hcat(u...)', width = 2.0, linetype = :steppost)
