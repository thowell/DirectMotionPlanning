struct StageConstraint
    p::Int
    info::Dict
end

ConstraintSet = Vector{StageConstraint}

struct StageConstraints <: Constraints
    con::ConstraintSet
    data::ConstraintsData
    T::Int
end

c!(a, cons::StageConstraints, x, u, t) = nothing

function constraints!(cons::StageConstraints, x, u)
    T = cons.T

    for t = 1:T-1
        c!(cons.data.c[t], cons, x[t], u[t], t)
    end

    c!(cons.data.c[T], cons, x[T], nothing, T)
end

function constraint_violation(cons::StageConstraints; norm_type = Inf)
    T = cons.T
    c_max = 0.0

    for t = 1:T
        c_viol = copy(cons.data.c[t])

        # find inequality constraints
        if haskey(cons.con[t].info, :inequality)
            for i in cons.con[t].info[:inequality]
                c_viol[i] = max.(0.0, cons.data.c[t][i])
            end
        end

        c_max = max(c_max, norm(c_viol, norm_type))
    end

    return c_max
end

function constraint_violation(cons::StageConstraints, x, u; norm_type = Inf)
    constraints!(cons, x, u)
    constraint_violation(cons, norm_type = norm_type)
end
