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
        # fw(z) = fd(model, x̄[t], ū[t], z, h, t)

        data.dyn_deriv.fx[t] .= ForwardDiff.jacobian(fx, x̄[t])
        data.dyn_deriv.fu[t] .= ForwardDiff.jacobian(fu, ū[t])
        # data.dyn_deriv.fw[t] = ForwardDiff.jacobian(fw, w[t])
    end
end

function objective_derivatives!(obj::StageCosts, data::ModelData)
    x̄ = data.x̄
    ū = data.ū
    T = data.T
    model = data.model
    n = model.n
    m = model.m

    for t = 1:T-1
        gx(z) = g(obj, z, ū[t], t)
        gu(z) = g(obj, x̄[t], z, t)
        gz(z) = g(obj, z[1:n], z[n .+ (1:m)], t)

        data.obj_deriv.gx[t] .= ForwardDiff.gradient(gx, x̄[t])
        data.obj_deriv.gu[t] .= ForwardDiff.gradient(gu, ū[t])
        data.obj_deriv.gxx[t] .= ForwardDiff.hessian(gx, x̄[t])
        data.obj_deriv.guu[t] .= ForwardDiff.hessian(gu, ū[t])
        data.obj_deriv.gux[t] .= ForwardDiff.hessian(gz,
            [x̄[t]; ū[t]])[n .+ (1:m), 1:n]
    end

    gxT(z) = g(obj, z, nothing, T)

    data.obj_deriv.gx[T] .= ForwardDiff.gradient(gxT, x̄[T])
    data.obj_deriv.gxx[T] .= ForwardDiff.hessian(gxT, x̄[T])
end

function constraints_derivatives!(cons::StageConstraints, data::ModelData)
    x̄ = data.x̄
    ū = data.ū
    T = data.T

    for t = 1:T-1
        c = cons.data.c[t]
        cx!(a, z) = c!(a, cons, z, ū[t], t)
        cu!(a, z) = c!(a, cons, x̄[t], z, t)

        ForwardDiff.jacobian!(cons.data.cx[t], cx!, c, x̄[t])
        ForwardDiff.jacobian!(cons.data.cu[t], cu!, c, ū[t])
    end

    c = cons.data.c[T]
    cxT!(a, z) = c!(a, cons, z, nothing, T)
    ForwardDiff.jacobian!(cons.data.cx[T], cxT!, c, x̄[T])
end

function objective_derivatives!(obj::AugmentedLagrangianCosts, data::ModelData)
    gx = data.obj_deriv.gx
    gu = data.obj_deriv.gu
    gxx = data.obj_deriv.gxx
    guu = data.obj_deriv.guu
    gux = data.obj_deriv.gux

    c = obj.cons.data.c
    cx = obj.cons.data.cx
    cu = obj.cons.data.cu
    ρ = obj.ρ
    λ = obj.λ
    a = obj.a

    T = data.T
    model = data.model

    objective_derivatives!(obj.costs, data)
    constraints_derivatives!(obj.cons, data)

    for t = 1:T-1
        gx[t] .+= cx[t]' * (λ[t] + ρ * a[t] .* c[t])
        gu[t] .+= cu[t]' * (λ[t] + ρ * a[t] .* c[t])
        gxx[t] .+= ρ * cx[t]' * Diagonal(a[t]) * cx[t]
        guu[t] .+= ρ * cu[t]' * Diagonal(a[t]) * cu[t]
        gux[t] .+= ρ * cu[t]' * Diagonal(a[t]) * cx[t]
    end

    gx[T] .+= cx[T]' * (λ[T] + ρ * a[T] .* c[T])
    gxx[T] .+= ρ * cx[T]' * Diagonal(a[T]) * cx[T]
end

function derivatives!(m_data::ModelData)
    dynamics_derivatives!(m_data)
    objective_derivatives!(m_data.obj, m_data)
end
