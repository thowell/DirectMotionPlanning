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

        # data.dyn_deriv.fx[t] .= ForwardDiff.jacobian(fx, x̄[t])
        # data.dyn_deriv.fu[t] .= ForwardDiff.jacobian(fu, ū[t])
        # # data.dyn_deriv.fw[t] = ForwardDiff.jacobian(fw, w[t])

		ForwardDiff.jacobian!(data.dyn_deriv.fx[t], fx, x̄[t])
        ForwardDiff.jacobian!(data.dyn_deriv.fu[t], fu, ū[t])
        # ForwardDiff.jacobian!(data.dyn_deriv.fw[t], fw, w[t])
    end
end

function objective_derivatives!(obj::StageCosts, data::ModelData)
    x̄ = data.x̄
    ū = data.ū
    T = data.T
    model = data.model
    n = data.n
    m = data.m

    for t = 1:T-1
		# println("time step $t")
		if obj.cost[t] isa QuadraticCost
			data.obj_deriv.gx[t] .= 2.0 * obj.cost[t].Q * x̄[t] + obj.cost[t].q
			data.obj_deriv.gu[t] .= 2.0 * obj.cost[t].R * ū[t] + obj.cost[t].r
			data.obj_deriv.gxx[t] .= 2.0 * obj.cost[t].Q
			data.obj_deriv.guu[t] .= 2.0 * obj.cost[t].R
			data.obj_deriv.gux[t] .= 0.0
		else
	        gx(z) = g(obj, z, ū[t], t)
	        gu(z) = g(obj, x̄[t], z, t)
	        gz(z) = g(obj, z[1:n[t]], z[n[t] .+ (1:m[t])], t)

	        ForwardDiff.gradient!(data.obj_deriv.gx[t], gx, x̄[t])
	        ForwardDiff.gradient!(data.obj_deriv.gu[t], gu, ū[t])
			ForwardDiff.hessian!(data.obj_deriv.gxx[t], gx, x̄[t])
	        ForwardDiff.hessian!(data.obj_deriv.guu[t], gu, ū[t])
	        data.obj_deriv.gux[t] .= ForwardDiff.hessian(gz,
	            [x̄[t]; ū[t]])[n[t] .+ (1:m[t]), 1:n[t]]
		end
    end

	if obj.cost[T] isa QuadraticCost
		data.obj_deriv.gx[T] .= 2.0 * obj.cost[T].Q * x̄[T] + obj.cost[T].q
		data.obj_deriv.gxx[T] .= 2.0 * obj.cost[T].Q
	else
	    gxT(z) = g(obj, z, nothing, T)
	    ForwardDiff.gradient!(data.obj_deriv.gx[T], gxT, x̄[T])
	    ForwardDiff.hessian!(data.obj_deriv.gxx[T], gxT, x̄[T])
	end
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
        gx[t] .+= cx[t]' * (λ[t] + ρ[t] .* a[t] .* c[t])
        gu[t] .+= cu[t]' * (λ[t] + ρ[t] .* a[t] .* c[t])
        gxx[t] .+= cx[t]' * Diagonal(ρ[t] .* a[t]) * cx[t]
        guu[t] .+= cu[t]' * Diagonal(ρ[t] .* a[t]) * cu[t]
        gux[t] .+= cu[t]' * Diagonal(ρ[t] .* a[t]) * cx[t]
    end

    gx[T] .+= cx[T]' * (λ[T] + ρ[T] .* a[T] .* c[T])
    gxx[T] .+= cx[T]' * Diagonal(ρ[T] .* a[T]) * cx[T]
end

function derivatives!(m_data::ModelData)
    dynamics_derivatives!(m_data)
    objective_derivatives!(m_data.obj, m_data)
end
