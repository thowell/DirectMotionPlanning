function rollout(model, x1, u, w, h, T)
    x_hist = [x1]

    for t = 1:T-1
        push!(x_hist, fd(model, x_hist[end], u[t], w[t], h, t))
    end

    return x_hist
end

function rollout(model, K, k, x̄, ū, w, h, T; α = 1.0)
    x_hist = [x̄[1]]
    u_hist = []
    for t = 1:T-1
        push!(u_hist, ū[t] + K[t] * (x_hist[end] - x̄[t]) + α * k[t])
        push!(x_hist, fd(model, x_hist[end], u_hist[end], w[t], h, t))
    end
    return x_hist, u_hist
end
