function models(model, obj, x̄, ū, w, h, T)
	models_data = ModelData[]

	for i = 1:N
		m_data = model_data(model, obj, w[i], h, T)
		m_data.x̄ .= x̄[i]
		m_data.ū .= ū[i]

		push!(models_data, m_data)
	end

	return models_data
end

function objective(data::ModelsData; mode = :nominal)
	N = length(data)
	J = 0.0

	for i = 1:N
		J += objective(data[i], mode = mode)
	end

	return J / N
end

function derivatives!(data::ModelsData)
	N = length(data)

	for i = 1:N
		derivatives!(data[i])
	end
end

function backward_pass!(p_data::PolicyData, models_data::ModelsData)
	N = length(models_data)
    T = models_data[1].T

    fx =  [m_data.dyn_deriv.fx for m_data in models_data]
    fu =  [m_data.dyn_deriv.fu for m_data in models_data]
	fw =  [m_data.dyn_deriv.fw for m_data in models_data]
    gx =  [m_data.obj_deriv.gx for m_data in models_data]
    gu =  [m_data.obj_deriv.gu for m_data in models_data]
    gxx = [m_data.obj_deriv.gxx for m_data in models_data]
    guu = [m_data.obj_deriv.guu for m_data in models_data]
    gux = [m_data.obj_deriv.gux for m_data in models_data]

	w = [m_data.w for m_data in models_data]

    # policy
    K = p_data.K
    k = p_data.k

    # value function approximation
    P = p_data.P
    p = p_data.p

    # state-action value function approximation
    Qx = p_data.Qx
    Qu = p_data.Qu
    Qxx = p_data.Qxx
    Quu = p_data.Quu
    Qux = p_data.Qux

    # terminal value function
    P[T] = sum([gxx[i][T] for i = 1:N]) ./ N
    p[T] = sum([gx[i][T] for i = 1:N]) ./ N

    for t = T-1:-1:1
		# println(fw[1][t])
		# println(w[1][t])
		# println(P[t+1] * fw[1][t] * w[1][t])
		# println(sum([gx[i][t] + fx[i][t]' * (p[t+1] + P[t+1] * fw[i][t] * w[i][t]) for i = 1:N]) ./ N)
		# println(Qx[t])
        Qx[t] =  sum([gx[i][t] + fx[i][t]' * (p[t+1] + P[t+1] * fw[i][t] * w[i][t]) for i = 1:N]) ./ N
        Qu[t] =  sum([gu[i][t] + fu[i][t]' * (p[t+1] + P[t+1] * fw[i][t] * w[i][t]) for i = 1:N]) ./ N
        Qxx[t] = sum([gxx[i][t] + fx[i][t]' * P[t+1] * fx[i][t] for i = 1:N]) ./ N
        Quu[t] = sum([guu[i][t] + fu[i][t]' * P[t+1] * fu[i][t] for i = 1:N]) ./ N
        Qux[t] = sum([gux[i][t] + fu[i][t]' * P[t+1] * fx[i][t] for i = 1:N]) ./ N

        K[t] = -1.0 * Quu[t] \ Qux[t]
        k[t] = -1.0 * Quu[t] \ Qu[t]

        P[t] =  Qxx[t] + K[t]' * Quu[t] * K[t] + K[t]' * Qux[t] + Qux[t]' * K[t]
        p[t] =  Qx[t] + K[t]' * Quu[t] * k[t] + K[t]' * Qu[t] + Qux[t]' * k[t]
    end
end

function lagrangian_gradient!(s_data::SolverData, p_data::PolicyData, m_data::ModelsData)
	lagrangian_gradient!(s_data, p_data,
        m_data[1].model.n, m_data[1].model.m, m_data[1].T)
end

function forward_pass!(p_data::PolicyData, m_data::ModelsData, s_data::SolverData;
    max_iter = 100)

	N = length(m_data)

	# gradient of Lagrangian
	lagrangian_gradient!(s_data, p_data, m_data)

    # reset solver status
    s_data.status = true

    # line search with rollout
    α = 1.0
    iter = 1
    while true
        iter > max_iter && (@error "forward pass failure", break)

        J = Inf
		i = 1

		while i <= N
			iter > max_iter && (@error "forward pass failure", break)

	        try
	            rollout!(p_data, m_data[i], α = α)
				# @show objective(m_data[i].obj, m_data[i].x̄, m_data[i].ū)
	            # @show objective(m_data[i].obj, m_data[i].x, m_data[i].u)
	            Δz!(m_data[i])
				i += 1
	        catch
	            @warn "rollout failure (model $i)"
				fill!(m_data[i].z, 0.0)
				α *= 0.5
				iter += 1
				i = 1
	        end
		end

		J = objective(m_data, mode = :current)
		println("J_prev: $(s_data.obj)")
		println("J     : $(J)")
		println("iter: $iter")
		if true#J < s_data.obj + 0.001 * α * s_data.gradient' * (sum([m.z for m in models_data]) ./ N)
            # update nominal
			# set nominal trajectories
			# x_ref = [sum([m.x̄[t] for m in m_data]) ./ N for t = 1:T]
			# u_ref = [sum([m.ū[t] for m in m_data]) ./ N for t = 1:T-1]

			for i = 1:N
	            m_data[i].x̄ .= deepcopy(m_data[i].x)
	            m_data[i].ū .= deepcopy(m_data[i].u)
				# m_data[i].x̄ .= x_ref
	            # m_data[i].ū .= u_ref
			end
            s_data.obj = J
            s_data.status = true
            break
        else
            α *= 0.5
            iter += 1
        end
    end
end
