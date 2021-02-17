"""
    simulate linear feedback policy
"""
function simulate_linear_feedback(
        model,
        K,
        x̄, ū,
        x_ref, u_ref,
        Q, R,
        T_sim, Δt,
        x1,
        w;
        ul = -Inf * ones(length(ū[1])),
        uu = Inf * ones(length(ū[1])))

    T = length(x̄)
    times = [(t - 1) * Δt for t = 1:T-1]
    tf = Δt * (T-1)
    t_sim = range(0, stop = tf, length = T_sim)
    t_ctrl = range(0, stop = tf, length = T)
    dt_sim = tf / (T_sim - 1)

    x_sim = [x1]
    u_sim = []

    J = 0.0
    Jx = 0.0
    Ju = 0.0

    A_state = Array(hcat(x̄...))
    A_ctrl = Array(hcat(ū...))

    for tt = 1:T_sim-1
        t = t_sim[tt]
        k = searchsortedlast(times, t)
        x = x_sim[end]

        x_cubic = Array(zero(x̄[1]))

        for i = 1:length(x̄[1])
            interp_cubic = CubicSplineInterpolation(t_ctrl, A_state[i,:])
            x_cubic[i] = interp_cubic(t)
        end

        u = ū[k] + K[k] * (x - x_cubic)

        # clip controls
        u = max.(u, ul)
        u = min.(u, uu)

        push!(x_sim, fd(model, x, u, w[k], dt_sim, tt))
        push!(u_sim, u)

        J += (x_sim[end] - x_ref[k])' * Q[k + 1] * (x_sim[end] - x_ref[k])
        J += (u_sim[end] - u_ref[k])' * R[k] * (u_sim[end] - u_ref[k])
        Jx += (x_sim[end] - x_ref[k])' * Q[k + 1] * (x_sim[end] - x_ref[k])
        Ju += (u_sim[end] - u_ref[k])' * R[k] * (u_sim[end] - u_ref[k])
    end

    return x_sim, u_sim, J / (T_sim - 1), Jx / (T_sim - 1), Ju / (T_sim - 1)
end


function simulate_policy(
        model,
        θ,
        x_ref, u_ref,
        Q, R,
        T_sim, Δt,
        x1,
        w;
        ul = -Inf * ones(length(ū[1])),
        uu = Inf * ones(length(ū[1])))

    T = length(x̄)
    times = [(t - 1) * Δt for t = 1:T-1]
    tf = Δt * (T-1)
    t_sim = range(0, stop = tf, length = T_sim)
    t_ctrl = range(0, stop = tf, length = T)
    dt_sim = tf / (T_sim - 1)

    x_sim = [x1]
    u_sim = []

    J = 0.0
    Jx = 0.0
    Ju = 0.0

    # A_state = Array(hcat(x̄...))
    # A_ctrl = Array(hcat(ū...))

    for tt = 1:T_sim-1
        t = t_sim[tt]
        k = searchsortedlast(times, t)
        x = x_sim[end]

        # x_cubic = Array(zero(x̄[1]))
        #
        # for i = 1:length(x̄[1])
        #     interp_cubic = CubicSplineInterpolation(t_ctrl, A_state[i,:])
        #     x_cubic[i] = interp_cubic(t)
        # end

        # # clip controls
        # u = max.(u, ul)
        # u = min.(u, uu)

        push!(x_sim, fd(model, x, [zeros(model.m); θ], w[k], dt_sim, tt))
        push!(u_sim, policy(θ, x, nothing, model.n, model.m))

        J += (x_sim[end] - x_ref[k])' * Q[k + 1] * (x_sim[end] - x_ref[k])
        J += (u_sim[end] - u_ref[k])' * R[k] * (u_sim[end] - u_ref[k])
        Jx += (x_sim[end] - x_ref[k])' * Q[k + 1] * (x_sim[end] - x_ref[k])
        Ju += (u_sim[end] - u_ref[k])' * R[k] * (u_sim[end] - u_ref[k])
    end

    return x_sim, u_sim, J / (T_sim - 1), Jx / (T_sim - 1), Ju / (T_sim - 1)
end
