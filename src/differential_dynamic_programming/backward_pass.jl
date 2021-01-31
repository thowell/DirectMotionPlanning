function backward_pass!(p_data::PolicyData, m_data::ModelData)
    T = m_data.T
    fx = m_data.dyn_deriv.fx
    fu = m_data.dyn_deriv.fu
    gx = m_data.obj_deriv.gx
    gu = m_data.obj_deriv.gu
    gxx = m_data.obj_deriv.gxx
    guu = m_data.obj_deriv.guu
    gux = m_data.obj_deriv.gux

    # policy
    K = p_data.K
    k = p_data.k

    # value function approximation
    P = p_data.P
    p = p_data.p

    # state-action value function approximation
    Qx = p_data.Qx
    Qu = p_data.Qu
    Qxx = p_data.Qxx
    Quu = p_data.Quu
    Qux = p_data.Qux

    # terminal value function
    P[T] = gxx[T]
    p[T] =  gx[T]

    for t = T-1:-1:1
        Qx[t] = gx[t] + fx[t]' * p[t+1]
        Qu[t] = gu[t] + fu[t]' * p[t+1]
        Qxx[t] = gxx[t] + fx[t]' * P[t+1] * fx[t]
        Quu[t] = guu[t] + fu[t]' * P[t+1] * fu[t]
        Qux[t] = gux[t] + fu[t]' * P[t+1] * fx[t]

        K[t] = -1.0 * Quu[t] \ Qux[t]
        k[t] = -1.0 * Quu[t] \ Qu[t]

        P[t] =  Qxx[t] + K[t]' * Quu[t] * K[t] + K[t]' * Qux[t] + Qux[t]' * K[t]
        p[t] =  Qx[t] + K[t]' * Quu[t] * k[t] + K[t]' * Qu[t] + Qux[t]' * k[t]
    end
end
