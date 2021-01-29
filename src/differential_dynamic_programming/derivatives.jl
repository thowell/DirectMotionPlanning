function dynamics_derivatives!(data::ModelData)
    x̄ = data.x̄
    ū = data.ū
    w = data.w
    h = data.h
    T = data.T
    model = data.model

    for t = 1:T-1
        fx(z) = fd(model, z, ū[t], w[t], h, t)
        fu(z) = fd(model, x̄[t], z, w[t], h, t)

        data.dyn_deriv.fx[t] = ForwardDiff.jacobian(fx, x̄[t])
        data.dyn_deriv.fu[t] = ForwardDiff.jacobian(fu, ū[t])
    end
end

function objective_derivatives!(data::ModelData)
    x̄ = data.x̄
    ū = data.ū
    w = data.w
    h = data.h
    T = data.T
    model = data.model

    for t = 1:T-1
        gx(z) = g(obj, z, ū[t], t)
        gu(z) = g(obj, x̄[t], z, t)

        data.obj_deriv.gx[t] = ForwardDiff.gradient(gx, x̄[t])
        data.obj_deriv.gu[t] = ForwardDiff.gradient(gu, ū[t])
        data.obj_deriv.gxx[t] = ForwardDiff.hessian(gx, x̄[t])
        data.obj_deriv.guu[t] = ForwardDiff.hessian(gu, ū[t])
    end

    gxT(z) = g(obj, z, nothing, T)

    data.obj_deriv.gx[T] = ForwardDiff.gradient(gxT, x̄[T])
    data.obj_deriv.gxx[T] = ForwardDiff.hessian(gxT, x̄[T])
end

function derivatives!(m_data::ModelData)
    dynamics_derivatives!(m_data)
    objective_derivatives!(m_data)
end
