function rollout!(p_data::PolicyData, m_data::ModelData; α = 1.0)
    x = m_data.x
    u = m_data.u
    w = m_data.w
    h = m_data.h
    T = m_data.T
    x̄ = m_data.x̄
    ū = m_data.ū
    model = m_data.model

    K = p_data.K
    k = p_data.k

    # initial state
    x[1] = copy(x̄[1])

    # rollout
    for t = 1:T-1
        u[t] = ū[t] + K[t] * (x[t] - x̄[t]) + α * k[t]
        x[t+1] = fd(model, x[t], u[t], w[t], h, t)
    end
end

function rollout(model, x1, u, w, h, T)
    x_hist = [x1]

    for t = 1:T-1
        push!(x_hist, fd(model, x_hist[end], u[t], w[t], h, t))
    end

    return x_hist
end
