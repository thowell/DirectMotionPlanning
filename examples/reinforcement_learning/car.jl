# Model
include_model("car")

# Horizon
T = 11

# Time step
tf = 1.0
h = tf / (T - 1)

# Bounds

# ul <= u <= uu

# Initial and final states
x1 = [0.0; 0.0; 0.0]
xT = [0.0; 1.0; 0.0]
xl, xu = state_bounds(model, T, x1 = x1)#, xT = xT)

# Objective
obj = quadratic_tracking_objective(
        [(t < T ? Diagonal(0.0 * ones(model.n))
            : Diagonal(1000.0 * ones(model.n))) for t = 1:T],
        [Diagonal(0.0 * ones(model.m)) for t = 1:T-1],
        [xT for t = 1:T], [zeros(model.m) for t = 1:T])

# Problem
prob = trajectory_optimization_problem(
           model,
           obj,
           T,
           h = h,
           xl = xl,
           xu = xu)

# Trajectory initialization
x0 = linear_interpolation(x1, xT, T) # linear interpolation on state
u0 = [0.001 * ones(model.m) for t = 1:T-1]

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

# Solve nominal problem
@time z̄, info = solve(prob, copy(z0))
x̄, ū = unpack(z̄, prob)
@show x̄[end]

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
visualize!(vis, model, x̄, Δt = h, traj = true)

# learn a policy

function policy(model, x, θ)
    n = model.n
    m = model.m

    W1 = reshape(θ[1:(n * n)], n, n)
    b1 = θ[n * n .+ (1:n)]

    W2 = reshape(θ[n * n + n .+ (1:(m * n))], m, n)
    b2 = θ[n * n + n + m * n .+ (1:m)]

    W3 = reshape(θ[n * n + n + m * n + m .+ (1:(m * m))], m, m)

    z1 = tanh.(W1 * x + b1)
    z2 = W3 * tanh.(W2 * z1 + b2)

    # W1 = reshape(θ[1:(m * n)], m, n)
    # b1 = θ[m * n .+ (1:m)]
    #
    # z2 = W1 * x + b1

    return z2
end

function objective(model, x, u, T)
    J = 0.0

    for t = 1:T
        if t < T
            J += 0.01 * (x[:, t] - xT)' * (x[:, t] - xT)
        else
            J += (x[:, T] - xT)' *Diagonal([1.0, 10.0, 1.0]) * (x[:, T] - xT)
        end
    end

    return J
end

function rollout(model, x1, θ, T, h)
    n = model.n
    m = model.m

    x = zeros(eltype(θ), n, T)
    x[:, 1] = x1

    u = zeros(eltype(θ), m, T-1)
    u[:, 1] = policy(model, x[:, 1], θ)

    for t = 1:T-1
        u[:, t] = policy(model, x[:, t], θ)
        x[:, t+1] = fd(model, x[:, t], u[:, t], 0.01 * randn(model.d), h, t)
    end

    J = objective(model, x, u, T)

    return x, u, J
end

function solve(model, x1)
    n = model.n
    m = model.m

    # policy parameters
    p = n * n + n + m * n + m + m * m
    θ = 0.1 * randn(p)

    # rewards
    r_hist = []

    # samples
    N = length(x1)

    α = 1.0e-3
    for i = 1:1e5
        function reward(z)
            J = sum([rollout(model, _x1, z, T, h)[3] for _x1 in x1])
            return J / N
        end

        r = reward(θ)
        push!(r_hist, r)

        if i % 1000 == 0
            println("reward (iter = $i): $(r_hist[end])")
        end
        ∇r = ForwardDiff.gradient(reward, θ)

        # println("gradient $(∇r)")

        θ .-= α * ∇r
    end

    return θ
end

s1 = resample(x1, Diagonal(0.5 * ones(n)), 1.0)
θ = solve(model, [x1, s1...])
s1[4]
_x̄, ū, J = rollout(model, x1, θ, T, h)
x̄ = [_x̄[:, t] for t = 1:T]
visualize!(vis, model, x̄, Δt = h, traj = true)
