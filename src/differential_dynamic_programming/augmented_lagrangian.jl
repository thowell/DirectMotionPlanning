mutable struct AugmentedLagrangianCosts{T} <: Objective
    costs::StageCosts
    cons::Constraints
    ρ::T       # penalty
    λ::Vector  # dual estimates
    a::Vector # active set
end

function augmented_lagrangian(costs::StageCosts, cons::Constraints; ρ = 0.0)
    λ = [zeros(cons.con[t].p) for t = 1:cons.T]
    a = [ones(cons.con[t].p) for t = 1:cons.T]
    AugmentedLagrangianCosts(costs, cons, ρ, λ, a)
end

function objective(obj::AugmentedLagrangianCosts, x, u)
    # costs
    J = objective(obj.costs, x, u)

    # constraints
    T = obj.cons.T
    c = obj.cons.data.c
    ρ = obj.ρ
    λ = obj.λ
    a = obj.a

    constraints!(obj.cons, x, u)
    active_set!(a, obj.cons, λ)

    for t = 1:T
        J += λ[t]' * c[t]
        J += 0.5 * ρ * c[t]' * Diagonal(a[t]) * c[t]
    end

    return J
end

function active_set!(a, cons::StageConstraints, λ)
    T = cons.T
    c = cons.data.c

    for t = 1:T
        # set all constraints active
        fill!(a[t], 1.0)

        # find inequality constraints
        if haskey(cons.con[t].info, :inequality)
            for i in cons.con[t].info[:inequality]
                # check active-set criteria
                (c[t][i] < 0.0 && λ[t][i] == 0.0) && (a[t][i] = 0.0)
            end
        end
    end
end

function augmented_lagrangian_update!(obj::AugmentedLagrangianCosts;
        s = 10.0, max_penalty = 1.0e8)
    # constraints
    T = obj.cons.T
    c = obj.cons.data.c
    ρ = obj.ρ
    λ = obj.λ
    a = obj.a

    for t = 1:T
        # dual estimate update
        λ[t] .+= ρ * c[t]

        # inequality projection
        if haskey(obj.cons.con[t].info, :inequality)
            idx = obj.cons.con[t].info[:inequality]
            λ[t][idx] = max.(0.0, view(λ[t], idx))
        end
    end

    obj.ρ = min(s * obj.ρ, max_penalty)
end
