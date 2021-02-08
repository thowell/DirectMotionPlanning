include_ddp()

# Model
include_model("cartpole")
n = model.n
m = model.m

# Time
T = 51
h = 0.1

# Initial conditions, controls, disturbances
x1 = [0.0, 0.0, 0.0, 0.0]
xT = [0.0, π, 0.0, 0.0] # goal state
ū = [1.0e-1 * ones(model.m) for t = 1:T-1]
w = [zeros(model.d) for t = 1:T-1]

# Rollout
x̄ = rollout(model, x1, ū, w, h, T)
# x̄ = linear_interpolation(x1, xT, T)

# Objective
Qt = Diagonal(1.0e-1 * ones(model.n))
Rt = Diagonal(1.0e-3 * ones(model.m))
QT = Diagonal(10.0 * ones(model.n))
obj = StageCosts([(t < T ? QuadraticCost(Qt, nothing, Rt, nothing)
    : QuadraticCost(QT, nothing, nothing, nothing)) for t = 1:T], T)

function g(obj::StageCosts, x, u, t) #TODO fix global: T, xT
    T = obj.T
    if t < T
        Q = obj.cost[t].Q
        R = obj.cost[t].R
        return (x - xT)' * Q * (x - xT) + u' * R * u
    elseif t == T
        Q = obj.cost[T].Q
        return (x - xT)' * Q * (x - xT)
    else
        return 0.0
    end
end

# Problem
prob = problem_data(model, obj, copy(x̄), copy(ū), w, h, T)

# Solve
@time ddp_solve!(prob,
    max_iter = 500, verbose = true)

x, u = current_trajectory(prob)
x̄, ū = nominal_trajectory(prob)


# Visualize
using Plots
plot(π * ones(T),
    width = 2.0, color = :black, linestyle = :dash)
plot!(hcat(x...)', width = 2.0, label = "")
plot(hcat(u..., u[end])',
    width = 2.0, linetype = :steppost)

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
# open(vis)
visualize!(vis, model, x, Δt = h)
