function solve(model, obj, x̄, ū, w, h, T;
    max_iter = 10,
    grad_tol = 1.0e-6,
    verbose = true)

    verbose && println("differential dynamic programming")

    # compute initial objective
    J = objective(obj, x̄, ū)
    verbose && println("    cost: $J")

    # compute initial derivatives
    fx, fu = dynamics_derivatives(model, x̄, ū, w, h, T)
    gx, gu, gxx, guu = objective_derivatives(obj, x̄, ū)

    for i = 1:max_iter
        K, k, P, p, ΔV, Qx, Qu, Qxx, Quu, Qux = backward_pass(fx, fu, gx, gu, gxx, guu)
        x̄, ū, fx, fu, gx, gu, gxx, guu, J = forward_pass(model, obj, K, k, x̄, ū, w, h, T, J)
        grad_norm = norm(gradient(fx, fu, gx, gu, p))
        verbose && println("    cost: $J\n    grad norm: $(grad_norm)")
        grad_norm < grad_tol && break
    end

    return x̄, ū#, K, k, P, p, J
end

"""
    gradient
        https://web.stanford.edu/class/ee363/lectures/lqr-lagrange.pdf
"""
function gradient(fx, fu, gx, gu, p)
    Lx = [(t < T ? gx[t] + fx[t]' * p[t+1] - p[t]
        : gx[T] - p[T]) for t = 1:T]
    Lu = [gu[t] + fu[t]' * p[t+1] for t = 1:T-1]

    return vcat(Lx..., Lu...)
end
