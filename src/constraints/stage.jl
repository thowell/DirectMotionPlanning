"""
    stage constraints
        con!(c, x, u)
"""

struct StageConstraints <: Constraints
    n
    ineq
    n_stage
    c_stage # in-place function con!(c, x, u)
    t_idx
end

function stage_constraints(c_stage, n_stage, ineq_stage, t_idx)
    N = length(t_idx)
    n = n_stage * N
    ineq = vcat([(t - 1) * n_stage .+ ineq_stage for t = 1:N]...)

    return StageConstraints(n, ineq, n_stage, c_stage, t_idx)
end


function constraints!(c, Z, con::StageConstraints, model, idx, h, T)
    for (i, t) in enumerate(con.t_idx)
        x = view(Z, idx.x[t])
        u = view(Z, idx.u[t])

        con.c_stage(view(c, (i - 1) * con.n_stage .+ (1:con.n_stage)), x, u, t)
    end
    nothing
end

function constraints_jacobian!(∇c, Z, con::StageConstraints, model, idx, h, T)
    c_tmp = zeros(con.n_stage)
    shift = 0

    for (i, t) in enumerate(con.t_idx)
        x = view(Z, idx.x[t])
        u = view(Z, idx.u[t])

        cx(c, y) = con.c_stage(c, y, u, t)
        cu(c, y) = con.c_stage(c, x, y, t)

        r_idx = (i - 1) * con.n_stage .+ (1:con.n_stage)

        c_idx = idx.x[t]
        len = length(r_idx) * length(c_idx)
        ForwardDiff.jacobian!(reshape(view(∇c, shift .+ (1:len)),
            con.n_stage, model.n),
            cx, c_tmp, x)
        shift += len

        c_idx = idx.u[t]
        len = length(r_idx) * length(c_idx)
        ForwardDiff.jacobian!(reshape(view(∇c, shift .+ (1:len)),
            con.n_stage, model.m),
            cu, c_tmp, u)
        shift += len
    end

    return nothing
end

function constraints_sparsity(con::StageConstraints, model, idx, T;
    shift_row = 0, shift_col = 0)

    row = []
    col = []

    shift = 0

    for (i, t) in enumerate(con.t_idx)

        r_idx = shift_row + (i - 1) * con.n_stage .+ (1:con.n_stage)

        c_idx = shift_col .+ idx.x[t]
        row_col!(row, col, r_idx, c_idx)

        c_idx = idx.u[t]
        row_col!(row, col, r_idx, c_idx)
    end

    return collect(zip(row, col))
end
