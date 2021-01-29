


function solve(model, obj, x̄, ū, w, h, T;
    max_iter = 10,
    grad_tol = 1.0e-6,
    verbose = true)

    println()
    verbose && println("differential dynamic programming")

    # allocate data
    m_data = model_data(model, obj, w, h, T)
    m_data.x̄ .= x̄
    m_data.ū .= ū

    p_data = policy_data(model, T)
    grad = zeros(model.n * T + model.m * (T - 1))

    # compute objective
    J = objective(obj, m_data.x̄, m_data.ū)

    # compute derivatives
    dynamics_derivatives!(m_data)
    objective_derivatives!(m_data)

    for i = 1:max_iter
        backward_pass!(p_data, m_data)
        J, status = forward_pass!(p_data, m_data, J)

        # check convergence
        gradient!(grad, p_data, m_data)
        grad_norm = norm(grad)
        verbose && println("    iter: $i
            cost: $J
            grad norm: $(grad_norm)")
        (!status || grad_norm < grad_tol) && break
    end

    return m_data.x̄, m_data.ū#, K, k, P, p, J
end

"""
    gradient
        https://web.stanford.edu/class/ee363/lectures/lqr-lagrange.pdf
"""
function gradient!(grad, p_data::PolicyData, m_data::ModelData)
    fx = m_data.dyn_deriv.fx
    fu = m_data.dyn_deriv.fu
    gx = m_data.obj_deriv.gx
    gu = m_data.obj_deriv.gu
    T = m_data.T
    n = m_data.model.n
    m = m_data.model.m

    p = p_data.p

    for t = 1:T
        idx_x = (t - 1) * n .+ (1:n)
        grad[idx_x] = (t < T ? gx[t] + fx[t]' * p[t+1] - p[t] : gx[T] - p[T])

        t == T && continue

        idx_u = n * T + (t - 1) * m .+ (1:m)
        grad[idx_u] = gu[t] + fu[t]' * p[t+1]
    end
end
