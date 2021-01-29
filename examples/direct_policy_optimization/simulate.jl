"""
    simulate linear feedback policy
"""
function simulate(
        model,
        policy, K,
        z_nom, u_nom,
        Q, R,
        T_sim, Δt,
        z0,
        w;
        _norm = 2,
        xl = -Inf * ones(length(z_nom[1])),
        xu = Inf * ones(length(z_nom[1])),
        ul = -Inf * ones(length(u_nom[1])),
        uu = Inf * ones(length(u_nom[1])),
        u_idx = 1:model.m)

    T = length(z_nom)
    times = [(t - 1) * Δt for t = 1:T-1]
    tf = Δt * (T-1)
    t_sim = range(0, stop = tf, length = T_sim)
    t_ctrl = range(0, stop = tf, length = T)
    dt_sim = tf / (T_sim - 1)

    z_rollout = [z0]
    u_rollout = []

    J = 0.0
    Jx = 0.0
    Ju = 0.0

    A_state = Array(hcat(z_nom...))
    A_ctrl = Array(hcat(u_nom...))

    for tt = 1:T_sim-1
        t = t_sim[tt]
        k = searchsortedlast(times, t)
        z = z_rollout[end]

        z_cubic = Array(zero(z_nom[1]))
        for i = 1:length(z_nom[1])
            interp_cubic = CubicSplineInterpolation(t_ctrl, A_state[i,:])
            z_cubic[i] = interp_cubic(t)
        end

        u = eval_policy(policy, K[k], z, z_cubic, u_nom[k])

        # clip controls
        u = max.(u, ul[u_idx])
        u = min.(u, uu[u_idx])

        push!(z_rollout, fd(model, z, u, w[:, tt], dt_sim, tt))
        push!(u_rollout, u)

        if _norm == 2
            J += (state_output(model, z_rollout[end]) - z_cubic)' * Q[k + 1] * (state_output(model, z_rollout[end]) - z_cubic)
            J += (u_rollout[end] - u_nom[k][u_idx])' * R[k][u_idx, u_idx] * (u_rollout[end] - u_nom[k][u_idx])
            Jx += (state_output(model, z_rollout[end]) - z_cubic)' * Q[k+1] * (state_output(model, z_rollout[end]) - z_cubic)
            Ju += (u_rollout[end] - u_nom[k][u_idx])' * R[k][u_idx, u_idx] * (u_rollout[end] - u_nom[k][u_idx])
        else
            J += norm(sqrt(Q[k + 1]) * (state_output(model, z_rollout[end]) - z_cubic), _norm)
            J += norm(sqrt(R[k][u_idx, u_idx]) * (u - u_nom[k][u_idx]), _norm)
            Jx += norm(sqrt(Q[k + 1]) * (state_output(model, z_rollout[end]) - z_cubic), _norm)
            Ju += norm(sqrt(R[k][u_idx, u_idx]) * (u - u_nom[k][u_idx]), _norm)
        end
    end

    return z_rollout, u_rollout, J / (T_sim - 1), Jx / (T_sim - 1), Ju / (T_sim - 1)
end
