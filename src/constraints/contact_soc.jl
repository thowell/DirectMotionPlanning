"""
	contact constraints
"""

struct ContactConstraints <: Constraints
	n
	ineq
end

function contact_constraints(model, T)
	n = model.nc * (T + 1) + model.nc * (T - 1) + 2 * model.nb * (T - 1) + 2 * (T - 1)
	ineq = con_ineq_contact(model, T)

	return ContactConstraints(n, ineq)
end

function constraints!(c, Z, con::ContactConstraints, model, idx, h, T)
	shift = 0

	# signed-distance function
	for t = 1:T
		x = view(Z, idx.x[t])

		if t == 1
			q = view(x, 1:model.nq)
			c[1:model.nc] = ϕ_func(model, q)
		end

		q⁺ = view(x, model.nq .+ (1:model.nq))
		c[model.nc + (t-1) * model.nc .+ (1:model.nc)] = ϕ_func(model, q⁺)
	end

	shift += model.nc * (T + 1)

	# friction cone
	for t = 1:T-1
		u = view(Z, idx.u[t])
		c[shift + (t - 1) * model.nc .+ (1:model.nc)] = friction_cone(model, u)
	end

	shift += model.nc * (T - 1)

	# maximum dissipation
	for t = 1:T-1
		x⁺ = view(Z, idx.x[t+1])
		u = view(Z, idx.u[t])
		c[shift + (t - 1) * 2 * model.nb .+ (1:2 * model.nb)] = maximum_dissipation(model,
			x⁺, u, h)
	end

	shift += 2 * model.nb * (T - 1)

	# impact complementarity
	for t = 1:T-1
		x⁺ = view(Z, idx.x[t + 1])
		u = view(Z, idx.u[t])

		q = view(x⁺, model.nq .+ (1:model.nq))
		s = view(u, model.idx_s)
		λ = view(u, model.idx_λ)

		c[shift + (t-1) * 1 + 1] = s[1] - (λ' * ϕ_func(model, q))[1]
	end

	shift += (T - 1)

	# friction cone complementarity
	for t = 1:T-1
		u = view(Z, idx.u[t])

		s = view(u, model.idx_s)
		ψ = view(u, model.idx_ψ)
		λ = view(u, model.idx_λ)
		b = view(u, model.idx_b)

		c[shift + (t-1) * 1 + 1] = s[1] - (ψ' * [model.μ * λ[1]; b])[1]
	end

	shift += (T - 1)

	# # friction parameter complementarity
	# for t = 1:T-1
	# 	u = view(Z, idx.u[t])
	#
	# 	s = view(u, model.idx_s)
	# 	b = view(u, model.idx_b)
	# 	η = view(u, model.idx_η)
	#
	# 	c[shift + (t-1) * 1 + 1] = s[1] - (η' * b)[1]
	# end
	#
	# shift += (T - 1)

    return nothing
end

function constraints_jacobian!(∇c, Z, con::ContactConstraints, model, idx, h, T)
	shift = 0
	c_shift = 0

	# signed-distance function
	ϕ(y) = ϕ_func(model, y)

	for t = 1:T
		x = view(Z, idx.x[t])

		if t == 1
			q = view(x, 1:model.nq)
			r_idx = 1:model.nc
			c_idx = idx.x[t][1:model.nq]
			len = length(r_idx) * length(c_idx)
			∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(ϕ, q))
			shift += len
		end

		q⁺ = view(x, model.nq .+ (1:model.nq))
		r_idx = model.nc + (t-1) * model.nc .+ (1:model.nc)
		c_idx = idx.x[t][model.nq .+ (1:model.nq)]
		len = length(r_idx) * length(c_idx)
		∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(ϕ, q⁺))
		shift += len
	end

	c_shift += model.nc * (T + 1)

	# friction cone
	fc(y) = friction_cone(model, y)
	for t = 1:T-1
		u = view(Z, idx.u[t])
		r_idx = c_shift + (t - 1) * model.nc .+ (1:model.nc)
		c_idx = idx.u[t]
		len = length(r_idx) * length(c_idx)
		∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(fc, u))
		shift += len
	end

	c_shift += model.nc * (T - 1)

	# maximum dissipation
	for t = 1:T-1
		x⁺ = view(Z, idx.x[t+1])
		u = view(Z, idx.u[t])

		mdx(y) = maximum_dissipation(model, y, u, h)
		mdu(y) = maximum_dissipation(model, x⁺, y, h)

		r_idx = c_shift + (t - 1) * 2 * model.nb .+ (1:2 * model.nb)
		c_idx = idx.x[t + 1]
		len = length(r_idx) * length(c_idx)
		∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(mdx, x⁺))
		shift += len

		c_idx = idx.u[t]
		len = length(r_idx) * length(c_idx)
		∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(mdu, u))
		shift += len
	end

	c_shift += 2 * model.nb * (T - 1)

	# impact complementarity
	for t = 1:T-1
		x⁺ = view(Z, idx.x[t + 1])
		u = view(Z, idx.u[t])

		q = view(x⁺, model.nq .+ (1:model.nq))
		s = view(u, model.idx_s)
		λ = view(u, model.idx_λ)

		cq(y) = s - [λ' * ϕ_func(model, y)]
		cu(y) = y[model.idx_s] - [y[model.idx_λ]' * ϕ_func(model, q)]

		print()
		r_idx = c_shift + (t-1) * 1 + 1
		c_idx = idx.x[t + 1][model.nq .+ (1:model.nq)]
		len = length(r_idx) * length(c_idx)
		∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(cq, q))
		shift += len

		c_idx = idx.u[t]
		len = length(r_idx) * length(c_idx)
		∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(cu, u))
		shift += len
	end

	c_shift += (T - 1)

	# friction cone complementarity
	for t = 1:T-1
		u = view(Z, idx.u[t])

		s = view(u, model.idx_s)
		ψ = view(u, model.idx_ψ)

		cu(y) = y[model.idx_s] - [y[model.idx_ψ]' * [model.μ * y[model.idx_λ][1]; y[model.idx_b]]]

		r_idx = c_shift + (t-1) * 1 + 1
		c_idx = idx.u[t]
		len = length(r_idx) * length(c_idx)
		∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(cu, u))
		shift += len
	end

	c_shift += (T - 1)

	return nothing
end

function constraints_sparsity(con::ContactConstraints, model, idx, T;
	shift_row = 0, shift_col = 0)

	row = []
    col = []

	con_shift = 0

	# signed-distance function
	for t = 1:T

		if t == 1
			r_idx = shift_row .+ (1:model.nc)
			c_idx = shift_col .+ idx.x[t][1:model.nq]
			row_col!(row, col, r_idx, c_idx)
		end

		r_idx = shift_row + model.nc + (t-1) * model.nc .+ (1:model.nc)
		c_idx = shift_col .+ idx.x[t][model.nq .+ (1:model.nq)]
		row_col!(row, col, r_idx, c_idx)
	end

	con_shift += model.nc * (T + 1)

	# friction cone
	for t = 1:T-1
		r_idx = shift_row + con_shift + (t - 1) * model.nc .+ (1:model.nc)
		c_idx = shift_col .+ idx.u[t]
		row_col!(row, col, r_idx, c_idx)
	end

	con_shift += model.nc * (T - 1)

	# maximum dissipation
	for t = 1:T-1
		r_idx = shift_row + con_shift + (t - 1) * 2 * model.nb .+ (1:2 * model.nb)
		c_idx = shift_col .+ idx.x[t + 1]
		row_col!(row, col, r_idx, c_idx)

		c_idx = shift_col .+ idx.u[t]
		row_col!(row, col, r_idx, c_idx)
	end

	con_shift += 2 * model.nb * (T - 1)

	# impact complementarity
	for t = 1:T-1
		r_idx = shift_row + con_shift + (t-1) * 1 + 1
		c_idx = shift_col .+ idx.x[t + 1][model.nq .+ (1:model.nq)]
		row_col!(row, col, r_idx, c_idx)

		c_idx = shift_col .+ idx.u[t]
		row_col!(row, col, r_idx, c_idx)
	end

	con_shift += (T - 1)

	# friction cone complementarity
	for t = 1:T-1
		r_idx = shift_row + con_shift + (t-1) * 1 + 1
		c_idx = shift_col .+ idx.u[t]
		row_col!(row, col, r_idx, c_idx)
	end

	con_shift += (T - 1)

    return collect(zip(row, col))
end

function con_ineq_contact(model, T)

	con_ineq = []
	shift = 0

	# signed-distance function
	push!(con_ineq, [i for i = 1:model.nc * (T + 1)])
	shift += model.nc * (T + 1)

	# friction cone
	push!(con_ineq, [i for i = shift .+ (1:model.nc * (T - 1))])
	shift += model.nc * (T - 1)

	# maximum dissipation
	# push!(con_ineq, [i for i = shift .+ (1:model.nb * (T - 1))])
	shift += 2 * model.nb * (T - 1)

	# impact complementarity
	push!(con_ineq, [i for i = shift .+ (1:(T - 1))])
	shift += (T - 1)

	# friction cone complementarity
	push!(con_ineq, [i for i = shift .+ (1:(T - 1))])
	shift += (T - 1)

    return vcat(con_ineq...)
end
