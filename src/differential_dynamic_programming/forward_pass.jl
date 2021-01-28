function forward_pass(model, obj, K, k, x̄, ū, w, h, T, J̄,
        max_iter = 10)

    # line search with rollout
    α = 1.0
    iter = 1
    while iter < max_iter
        x, u = rollout(model, K, k, x̄, ū, w, h, T, α = α)
        J = objective(obj, x, u)

        if J < J̄
            # update nominal
            x̄ = deepcopy(x)
            ū = deepcopy(u)
            J̄ = J
            break
        else
            α *= 0.5
            iter += 1
        end
    end

    # derivatives
    fx, fu = dynamics_derivatives(model, x̄, ū, w, h, T)
    gx, gu, gxx, guu = objective_derivatives(obj, x̄, ū)

    return x̄, ū, fx, fu, gx, gu, gxx, guu, J̄
end
