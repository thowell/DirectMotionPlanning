function dynamics_derivatives(model, x, u, w, h, T)
    fx_hist = []
    fu_hist = []

    for t = 1:T-1
        # Jacobians
        fx(z) = fd(model, z, u[t], w[t], h, t)
        fu(z) = fd(model, x[t], z, w[t], h, t)

        push!(fx_hist, ForwardDiff.jacobian(fx, x[t]))
        push!(fu_hist, ForwardDiff.jacobian(fu, u[t]))

    end

    return fx_hist, fu_hist
end

function objective_derivatives(obj, x, u)
    T = length(x)
    gx_hist = []
    gu_hist = []

    gxx_hist = []
    guu_hist = []

    for t = 1:T-1
        gx(z) = g(obj, z, u[t], t)
        gu(z) = g(obj, x[t], z, t)

        push!(gx_hist, ForwardDiff.gradient(gx, x[t]))
        push!(gu_hist, ForwardDiff.gradient(gu, u[t]))
        push!(gxx_hist, ForwardDiff.hessian(gx, x[t]))
        push!(guu_hist, ForwardDiff.hessian(gu, u[t]))
    end

    gx(z) = g(obj, z, nothing, T)

    push!(gx_hist, ForwardDiff.gradient(gx, x[T]))
    push!(gxx_hist, ForwardDiff.hessian(gx, x[T]))

    return gx_hist, gu_hist, gxx_hist, guu_hist
end
