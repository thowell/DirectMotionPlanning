function backward_pass(fx, fu, gx, gu, gxx, guu)
    T = length(gx)

    # policy
    K = []
    k = []

    # value function approximation
    P = []
    p = []
    ΔV = []

    # state-action value function approximation
    Qx = []
    Qu = []
    Qxx = []
    Quu = []
    Qux = []

    push!(P, gxx[T])
    push!(p, gx[T])

    for t = T-1:-1:1
        push!(Qx, gx[t] + fx[t]' * p[end])
        push!(Qu, gu[t] + fu[t]' * p[end])
        push!(Qxx, gxx[t] + fx[t]' * P[end] * fx[t])
        push!(Quu, guu[t] + fu[t]' * P[end] * fu[t])
        push!(Qux, fu[t]' * P[end] * fx[t])

        push!(K, -1.0 * Quu[end] \ Qux[end])
        push!(k, -1.0 * Quu[end] \ Qu[end])

        push!(P, Qxx[end] + K[end]' * Quu[end] * K[end]
            + K[end]' * Qux[end] + Qux[end]' * K[end])
        push!(p, Qx[end] + K[end]' * Quu[end] * k[end]
            + K[end]' * Qu[end] + Qux[end]' * k[end])
        push!(ΔV, (k[end]' * Qu[end], 0.5 * k[end]' * Quu[end] * k[end]))
    end

    return reverse(K), reverse(k), reverse(P), reverse(p), reverse(ΔV), reverse(Qx), reverse(Qu), reverse(Qxx), reverse(Quu), reverse(Qux)
end
