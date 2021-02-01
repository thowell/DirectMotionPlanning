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
