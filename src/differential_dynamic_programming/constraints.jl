abstract type StageConstraint end
abstract type StageConstraints end

c!(c, con::StageConstraint, x, u, t) = nothing

function constraints!(c_data::ConstraintsData, con::StageConstraints, x, u)
    T = con.T

    for t = 1:T-1
        c!(c_data.c[t], con[t], x[t], u[t], t)
    end

    c!(c_data.c[T], con[T], x[T], nothing, T)
end
