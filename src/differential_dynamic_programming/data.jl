"""
    Model Data
"""

struct DynamicsDerivativesData{X, U}
    fx::Vector{X}
    fu::Vector{U}
end

function dynamics_derivatives_data(model::Model, T)
    n = model.n
    m = model.m

    fx = [SMatrix{n, n}(zeros(n, n)) for t = 1:T-1]
    fu = [SMatrix{n, m}(zeros(n, m)) for t = 1:T-1]

    DynamicsDerivativesData(fx, fu)
end

struct ObjectiveDerivativesData{X, U, XX, UU}
    gx::Vector{X}
    gu::Vector{U}
    gxx::Vector{XX}
    guu::Vector{UU}
end

function objective_derivatives_data(model::Model, T)
    n = model.n
    m = model.m

    gx = [SVector{n}(ones(n)) for t = 1:T]
    gu = [SVector{m}(ones(m)) for t = 1:T-1]
    gxx = [SMatrix{n, n}(ones(n, n)) for t = 1:T]
    guu = [SMatrix{m, m}(ones(m, m)) for t = 1:T-1]

    ObjectiveDerivativesData(gx, gu, gxx, guu)
end

struct ModelData{X, U, D, S}
    # current trajectory
    x::Vector{X}
    u::Vector{U}

    # disturbance trajectory
    w::Vector{D}

    # time step
    h::S

    # horizon
    T

    # nominal trajectory
    x̄::Vector{X}
    ū::Vector{U}

    # dynamics model
    model::Model

    # objective
    obj::StageObjective

    # dynamics derivatives data
    dyn_deriv::DynamicsDerivativesData

    # objective derivatives data
    obj_deriv::ObjectiveDerivativesData
end

function model_data(model, obj, w, h, T)
    n = model.n
    m = model.m

    x = [SVector{n}(zeros(n)) for t = 1:T]
    u = [SVector{m}(zeros(m)) for t = 1:T-1]

    x̄ = [SVector{n}(zeros(n)) for t = 1:T]
    ū = [SVector{m}(zeros(m)) for t = 1:T-1]

    dyn_deriv = dynamics_derivatives_data(model, T)
    obj_deriv = objective_derivatives_data(model, T)

    ModelData(x, u, w, h, T, x̄, ū, model, obj, dyn_deriv, obj_deriv)
end

"""
    Policy Data
"""
struct PolicyData{N, M, NN, MM, MN}
    # policy
    K::Vector{MN}
    k::Vector{M}

    # value function approximation
    P::Vector{NN}
    p::Vector{N}
    ΔV::Vector

    # state-action value function approximation
    Qx::Vector{N}
    Qu::Vector{M}
    Qxx::Vector{NN}
    Quu::Vector{MM}
    Qux::Vector{MN}
end

function policy_data(model::Model, T)
    n = model.n
    m = model.m

    K = [SMatrix{m, n}(zeros(m, n)) for t = 1:T-1]
    k = [SVector{m}(zeros(m)) for t = 1:T-1]

    P = [SMatrix{n, n}(zeros(n, n)) for t = 1:T]
    p = [SVector{n}(zeros(n)) for t = 1:T]
    ΔV = [() for t = 1:T]

    Qx = [SVector{n}(zeros(n)) for t = 1:T-1]
    Qu = [SVector{m}(zeros(m)) for t = 1:T-1]
    Qxx = [SMatrix{n, n}(zeros(n, n)) for t = 1:T-1]
    Quu = [SMatrix{m, m}(zeros(m, m)) for t = 1:T-1]
    Qux = [SMatrix{m, n}(zeros(m, n)) for t = 1:T-1]

    PolicyData(K, k, P, p, ΔV, Qx, Qu, Qxx, Quu, Qux)
end
