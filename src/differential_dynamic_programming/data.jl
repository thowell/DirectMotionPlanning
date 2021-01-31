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

struct ObjectiveDerivativesData{X, U, XX, UU, UX}
    gx::Vector{X}
    gu::Vector{U}
    gxx::Vector{XX}
    guu::Vector{UU}
    gux::Vector{UX}
end

function objective_derivatives_data(model::Model, T)
    n = model.n
    m = model.m

    gx = [SVector{n}(ones(n)) for t = 1:T]
    gu = [SVector{m}(ones(m)) for t = 1:T-1]
    gxx = [SMatrix{n, n}(ones(n, n)) for t = 1:T]
    guu = [SMatrix{m, m}(ones(m, m)) for t = 1:T-1]
    gux = [SMatrix{m, n}(ones(m, n)) for t = 1:T-1]

    ObjectiveDerivativesData(gx, gu, gxx, guu, gux)
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
    T::Int

    # nominal trajectory
    x̄::Vector{X}
    ū::Vector{U}

    # dynamics model
    model::Model

    # objective
    obj::StageCosts

    # dynamics derivatives data
    dyn_deriv::DynamicsDerivativesData

    # objective derivatives data
    obj_deriv::ObjectiveDerivativesData

    # z = (x1...,xT,u1,...,uT-1) | Δz = (Δx1...,ΔxT,Δu1,...,ΔuT-1)
    z::Vector{S}
end

function model_data(model, obj, w, h, T)
    n = model.n
    m = model.m
    num_var = n * T + m * (T - 1)

    x = [SVector{n}(zeros(n)) for t = 1:T]
    u = [SVector{m}(zeros(m)) for t = 1:T-1]

    x̄ = [SVector{n}(zeros(n)) for t = 1:T]
    ū = [SVector{m}(zeros(m)) for t = 1:T-1]

    dyn_deriv = dynamics_derivatives_data(model, T)
    obj_deriv = objective_derivatives_data(model, T)

    z = zeros(num_var)

    ModelData(x, u, w, h, T, x̄, ū, model, obj, dyn_deriv, obj_deriv, z)
end

function Δz!(m_data::ModelData)
    for t = 1:T
        idx_x = (t - 1) * n .+ (1:n)
        m_data.z[idx_x] = m_data.x[t] - m_data.x̄[t]

        t == T && continue

        idx_u = n * T + (t - 1) * m .+ (1:m)
        m_data.z[idx_u] = m_data.u[t] - m_data.ū[t]
    end
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

    Qx = [SVector{n}(zeros(n)) for t = 1:T-1]
    Qu = [SVector{m}(zeros(m)) for t = 1:T-1]
    Qxx = [SMatrix{n, n}(zeros(n, n)) for t = 1:T-1]
    Quu = [SMatrix{m, m}(zeros(m, m)) for t = 1:T-1]
    Qux = [SMatrix{m, n}(zeros(m, n)) for t = 1:T-1]

    PolicyData(K, k, P, p, Qx, Qu, Qxx, Quu, Qux)
end

"""
    Solver Data
"""
mutable struct SolverData{T}
    obj::T              # objective value
    gradient::Vector{T} # Lagrangian gradient
    status::Bool        # solver status
end

function solver_data(model::Model, T)
    n = model.n
    m = model.m
    num_var = n * T + m * (T - 1)

    obj = Inf
    gradient = zeros(num_var)

    SolverData(obj, gradient, false)
end

"""
    Constraints Data
"""
struct ConstraintsData
    c
    cx
    cu
end

function constraints_data(model::Model, p::Vector, T::Int)
    n = model.n
    m = model.m
    c = [SVector{p[t]}(zeros(p[t])) for t = 1:T]
    cx = [SMatrix{p[t], n}(zeros(p[t], n)) for t = 1:T]
    cu = [SMatrix{p[t], m}(zeros(p[t], m)) for t = 1:T]

    ConstraintsData(c, cx, cu)
end
