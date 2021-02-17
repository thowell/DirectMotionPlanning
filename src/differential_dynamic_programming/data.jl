"""
    Model Data
"""

struct DynamicsDerivativesData{X, U, W}
    fx::Vector{X}
    fu::Vector{U}
	fw::Vector{W}
end

function dynamics_derivatives_data(model::Model, T;
	n = [model.n for t = 1:T],
	m = [model.m for t = 1:T-1],
	d = [model.d for t = 1:T-1])

	fx = [zeros(n[t + 1], n[t]) for t = 1:T-1]
    fu = [zeros(n[t + 1], m[t]) for t = 1:T-1]
	fw = [zeros(n[t + 1], d[t]) for t = 1:T-1]

    DynamicsDerivativesData(fx, fu, fw)
end

struct ObjectiveDerivativesData{X, U, XX, UU, UX}
    gx::Vector{X}
    gu::Vector{U}
    gxx::Vector{XX}
    guu::Vector{UU}
    gux::Vector{UX}
end

function objective_derivatives_data(model::Model, T;
	n = [model.n for t = 1:T],
	m = [model.m for t = 1:T-1])

	gx = [ones(n[t]) for t = 1:T]
    gu = [ones(m[t]) for t = 1:T-1]
    gxx = [ones(n[t], n[t]) for t = 1:T]
    guu = [ones(m[t], m[t]) for t = 1:T-1]
    gux = [ones(m[t], n[t]) for t = 1:T-1]

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
	n::Vector{Int}
	m::Vector{Int}
	d::Vector{Int}

    # objective
    obj::Objective

    # dynamics derivatives data
    dyn_deriv::DynamicsDerivativesData

    # objective derivatives data
    obj_deriv::ObjectiveDerivativesData

    # z = (x1...,xT,u1,...,uT-1) | Δz = (Δx1...,ΔxT,Δu1,...,ΔuT-1)
    z::Vector{S}
end

ModelsData = Vector{ModelData}

function model_data(model, obj, w, h, T;
	n = [model.n for t = 1:T],
	m = [model.m for t = 1:T-1],
	d = [model.d for t = 1:T-1])

    num_var = sum(n) + sum(m)

	x = [zeros(n[t]) for t = 1:T]
    u = [zeros(m[t]) for t = 1:T-1]

    x̄ = [zeros(n[t]) for t = 1:T]
    ū = [zeros(m[t]) for t = 1:T-1]

    dyn_deriv = dynamics_derivatives_data(model, T, n = n, m = m, d = d)
    obj_deriv = objective_derivatives_data(model, T, n = n, m = m)

    z = zeros(num_var)

    ModelData(x, u, w, h, T, x̄, ū, model, n, m, d, obj, dyn_deriv, obj_deriv, z)
end

function Δz!(m_data::ModelData)
	n = m_data.n
	m = m_data.m
	T = m_data.T

    for t = 1:T
        idx_x = (t == 1 ? 0 : (t - 1) * n[t-1]) .+ (1:n[t])
        m_data.z[idx_x] = m_data.x[t] - m_data.x̄[t]

        t == T && continue

        idx_u = sum(n) + (t == 1 ? 0 : (t - 1) * m[t-1]) .+ (1:m[t])
        m_data.z[idx_u] = m_data.u[t] - m_data.ū[t]
    end
end

function objective(data::ModelData; mode = :nominal)
    if mode == :nominal
        return objective(data.obj, data.x̄, data.ū)
    elseif mode == :current
        return objective(data.obj, data.x, data.u)
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

function policy_data(model::Model, T;
	n = [model.n for t = 1:T],
	m = [model.m for t = 1:T-1])

	K = [zeros(m[t], n[t]) for t = 1:T-1]
    k = [zeros(m[t]) for t = 1:T-1]

    P = [zeros(n[t], n[t]) for t = 1:T]
    p = [zeros(n[t]) for t = 1:T]

    Qx = [zeros(n[t]) for t = 1:T-1]
    Qu = [zeros(m[t]) for t = 1:T-1]
    Qxx = [zeros(n[t], n[t]) for t = 1:T-1]
    Quu = [zeros(m[t], m[t]) for t = 1:T-1]
    Qux = [zeros(m[t], n[t]) for t = 1:T-1]

    PolicyData(K, k, P, p, Qx, Qu, Qxx, Quu, Qux)
end

"""
    Solver Data
"""
mutable struct SolverData{T}
    obj::T              # objective value
    gradient::Vector{T} # Lagrangian gradient
	c_max::T            # maximum constraint violation

	α::T                # step length
    status::Bool        # solver status

	cache::Dict         #
end

cache = Dict(:obj => [], :gradient => [], :c_max => [], :α => [])

function solver_data(model::Model, T;
	n = [model.n for t = 1:T],
	m = [model.m for t = 1:T-1])

    num_var = sum(n) + sum(m)

    obj = Inf
	c_max = 0.0
    gradient = zeros(num_var)
	cache = Dict(:obj => [], :gradient => [], :c_max => [], :α => [])

    SolverData(obj, gradient, c_max, 1.0, false, cache)
end

function cache!(data::SolverData)
	push!(data.cache[:obj], data.obj)
	push!(data.cache[:gradient], data.gradient)
	push!(data.cache[:c_max], data.c_max)
	push!(data.cache[:α], data.α)
end

function objective!(s_data::SolverData, m_data::ModelData; mode = :nominal)
	if mode == :nominal
		s_data.obj = objective(m_data.obj, m_data.x̄, m_data.ū)
	elseif mode == :current
		s_data.obj = objective(m_data.obj, m_data.x, m_data.u)
	end

	if m_data.obj isa AugmentedLagrangianCosts
		s_data.c_max = constraint_violation(m_data.obj.cons,
			m_data.x, m_data.u,
			norm_type = Inf)
	end

	return s_data.obj
end

"""
    Problem Data
"""
struct ProblemData
	p_data
	m_data
	s_data
end

function problem_data(model::Model, obj::StageCosts, x̄, ū, w, h, T;
	n = [model.n for t = 1:T],
	m = [model.m for t = 1:T-1],
	d = [model.d for t = 1:T-1])

	# allocate policy data
    p_data = policy_data(model, T, n = n, m = m)

    # allocate model data
    m_data = model_data(model, obj, w, h, T, n = n, m = m, d = d)
    m_data.x̄ .= x̄
    m_data.ū .= ū

    # allocate solver data
    s_data = solver_data(model, T, n = n, m = m)

	ProblemData(p_data, m_data, s_data)
end

function problem_data(m_data::ModelsData;
	n = m_data[1].n,
	m = m_data[1].m)

	model = m_data[1].model
	T = m_data[1].T

	# allocate policy data
    p_data = policy_data(model, T, n = n, m = m)

    # allocate solver data
    s_data = solver_data(model, T, n = n, m = m)

	ProblemData(p_data, m_data, s_data)
end

function nominal_trajectory(prob::ProblemData)
	return prob.m_data.x̄, prob.m_data.ū
end

function current_trajectory(prob::ProblemData)
	return prob.m_data.x, prob.m_data.u
end

function problem_data(model, obj::StageCosts, con_set::ConstraintSet,
		x̄, ū, w, h, T;
		n = [model.n for t = 1:T],
		m = [model.m for t = 1:T-1],
		d = [model.d for t = 1:T-1])

	# constraints
	c_data = constraints_data(model, [c.p for c in con_set], T, n = n, m = m)
	cons = StageConstraints(con_set, c_data, T)

	# augmented Lagrangian
	obj_al = augmented_lagrangian(obj, cons)

	# allocate policy data
    p_data = policy_data(model, T, n = n, m = m)

    # allocate model data
    m_data = model_data(model, obj_al, w, h, T, n = n, m = m, d = d)
    m_data.x̄ .= x̄
    m_data.ū .= ū

    # allocate solver data
    s_data = solver_data(model, T, n = n, m = m)

	ProblemData(p_data, m_data, s_data)
end
