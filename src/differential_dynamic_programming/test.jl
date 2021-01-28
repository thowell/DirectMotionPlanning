include(joinpath(@__DIR__, "rollout.jl"))
include(joinpath(@__DIR__, "objective.jl"))
include(joinpath(@__DIR__, "derivatives.jl"))
include(joinpath(@__DIR__, "backward_pass.jl"))

include_model("double_integrator")

n = model.n
m = model.m

T = 10
h = 1.0

x1 = rand(model.n)
ū = [rand(model.m) for t = 1:T-1]
w = [zeros(model.d) for t = 1:T-1]

# rollout
x̄ = rollout(model, x1,
    [0.001 * rand(model.m) for t = 1:T-1],
    [zeros(model.d) for t = 1:T-1],
    h, T)

# objective
Q = Diagonal(ones(model.n))
R = Diagonal(ones(model.m))
g(x, u) = x' * Q * x + u' * R * u
gT(x) = x' * Q * x
obj = StageObjective([t < T ? g : gT for t = 1:T])
objective(obj, x̄, ū)

fx, fu = dynamics_derivatives(model, x̄, ū, w, h, T)
gx, gu, gxx, guu = objective_derivatives(obj, x̄, ū)

K, _k, P, p, ΔV, Qx, Qu, Qxx, Quu, Qux = backward_pass(fx, fu, gx, gu, gxx, guu)

x, u = rollout(model, K, _k, x1, x̄, ū, w, h, T, α = 1.0)

objective(obj, x, u)
