struct DynamicsConstraints <: Constraints
    n::Int
    ineq
    w
end

function dynamics_constraints(model, T;
    w = [zeros(model.d) for t = 1:T-1])

    n = model.n * (T - 1)
    ineq = (1:0)

    return DynamicsConstraints(n, ineq, w)
end

function constraints!(c, Z, con::DynamicsConstraints, model, idx, h, T)
    n = model.n
    nq = model.nq

    # note: x1 and xT constraints are handled as simple bound constraints
    #       e.g., x1 <= x <= x1, xT <= x <= xT

    for t = 1:T-1
        x = view(Z, idx.x[t])
        u = view(Z, idx.u[t])
        x⁺ = view(Z, idx.x[t + 1])

        q1 = view(x, 1:nq)
        q2 = view(x, nq .+ (1:nq))
        q2_next = view(x⁺, 1:nq)
        q3 = view(x⁺, nq .+ (1:nq))

        c[(t-1) * n .+ (1:nq)] = q3 - step(q1, q2, u, h, tol = 1.0e-5)[1]
        c[(t-1) * n + nq .+ (1:nq)] = q2_next - q2
    end

    return nothing
end

function constraints_jacobian!(∇c, Z, con::DynamicsConstraints, model, idx, h, T)
    n = model.n
    m = model.m
    nq = model.nq
    # note: x1 and xT constraints are handled as simple bound constraints
    #       e.g., x1 <= x <= x1, xT <= x <= xT

    shift = 0

    In = Diagonal(ones(model.nq))

    for t = 1:T-1
        x = view(Z, idx.x[t])
        u = view(Z, idx.u[t])
        x⁺ = view(Z, idx.x[t + 1])

        q1 = view(x, 1:nq)
        q2 = view(x, nq .+ (1:nq))
        q2_next = view(x⁺, 1:nq)
        q3 = view(x⁺, nq .+ (1:nq))
        _, _, _, Δq1, Δq2, Δu1, _ = step(q1, q2, u, h, tol = 1.0e-5)

        #c[(t-1) * n .+ (1:nq)] = q3 - step(q1, q2, u, h)[1]
        r_idx = (t-1) * n .+ (1:nq)

        c_idx = idx.x[t + 1][nq .+ (1:nq)]
        s = length(r_idx) * length(c_idx)
        ∇c[shift .+ (1:s)] = vec(In)
        shift += s

        c_idx = idx.x[t][1:nq]
        s = length(r_idx) * length(c_idx)
        ∇c[shift .+ (1:s)] = -1.0 * vec(Δq1)
        shift += s

        c_idx = idx.x[t][nq .+ (1:nq)]
        s = length(r_idx) * length(c_idx)
        ∇c[shift .+ (1:s)] = -1.0 * vec(Δq2)
        shift += s

        c_idx = idx.u[t]
        s = length(r_idx) * length(c_idx)
        ∇c[shift .+ (1:s)] = -1.0 * vec(Δu1)
        shift += s

        # c[(t-1) * n + nq .+ (1:nq)] = q2_next - q2
        r_idx = (t-1) * n + nq .+ (1:nq)

        c_idx = idx.x[t + 1][1:nq]
        s = length(r_idx) * length(c_idx)
        ∇c[shift .+ (1:s)] = vec(In)
        shift += s

        c_idx = idx.x[t][nq .+ (1:nq)]
        s = length(r_idx) * length(c_idx)
        ∇c[shift .+ (1:s)] = -1.0 * Diagonal(In)
        shift += s
    end

    return nothing
end

function constraints_sparsity(con::DynamicsConstraints, model, idx, T;
    shift_row = 0, shift_col = 0)
    n = model.n
    nq = model.nq
    row = []
    col = []

    for t = 1:T-1

        #c[(t-1) * n .+ (1:nq)] = q3 - step(q1, q2, u, h)[1]
        r_idx = (t-1) * n .+ (1:nq)

        c_idx = idx.x[t + 1][nq .+ (1:nq)]
        row_col!(row, col, shift_row .+ r_idx, shift_col .+ c_idx)

        c_idx = idx.x[t][1:nq]
        row_col!(row, col, shift_row .+ r_idx, shift_col .+ c_idx)

        c_idx = idx.x[t][nq .+ (1:nq)]
        row_col!(row, col, shift_row .+ r_idx, shift_col .+ c_idx)

        c_idx = idx.u[t]
        row_col!(row, col, shift_row .+ r_idx, shift_col .+ c_idx)

        # c[(t-1) * n + nq .+ (1:nq)] = q2_next - q2
        r_idx = (t-1) * n + nq .+ (1:nq)

        c_idx = idx.x[t + 1][1:nq]
        row_col!(row, col, shift_row .+ r_idx, shift_col .+ c_idx)

        c_idx = idx.x[t][nq .+ (1:nq)]
        row_col!(row, col, shift_row .+ r_idx, shift_col .+ c_idx)
    end

    return collect(zip(row, col))
end
