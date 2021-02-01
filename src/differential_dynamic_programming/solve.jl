function solve(model, obj::Objective, x̄, ū, w, h, T;
    max_iter = 10,
    grad_tol = 1.0e-5,
    verbose = true)

    println()
    verbose && println("differential dynamic programming")

    # allocate model data
    m_data = model_data(model, obj, w, h, T)
    m_data.x̄ .= x̄
    m_data.ū .= ū

    # allocate policy data
    p_data = policy_data(model, T)

    # allocate solver data
    s_data = solver_data(model, T)

    # compute objective
    s_data.obj = objective(m_data.obj, m_data.x̄, m_data.ū)

    for i = 1:max_iter
        # derivatives
        derivatives!(m_data)

        # backward pass
        backward_pass!(p_data, m_data)

        # forward pass
        forward_pass!(p_data, m_data, s_data)

        # check convergence
        grad_norm = norm(s_data.gradient, Inf)
        verbose && println("    iter: $i
            cost: $(s_data.obj)
            grad norm: $(grad_norm)")
        (!s_data.status || grad_norm < grad_tol) && break
    end

    return p_data, m_data, s_data
end

"""
    gradient of Lagrangian
        https://web.stanford.edu/class/ee363/lectures/lqr-lagrange.pdf
"""
function lagrangian_gradient!(s_data::SolverData, p_data::PolicyData, m_data::ModelData)
    T = m_data.T
    n = m_data.model.n
    m = m_data.model.m

    p = p_data.p
    Qx = p_data.Qx
    Qu = p_data.Qu

    for t = 1:T-1
        idx_x = (t - 1) * n .+ (1:n)
        s_data.gradient[idx_x] = Qx[t] - p[t]
        # NOTE: gradient wrt xT is satisfied implicitly

        idx_u = n * T + (t - 1) * m .+ (1:m)
        s_data.gradient[idx_u] = Qu[t]
    end
end

"""
    augmented Lagrangian solve
"""
function solve()
end
