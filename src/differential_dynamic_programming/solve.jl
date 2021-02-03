function solve(model, obj::Objective, x̄, ū, w, h, T;
    max_iter = 10,
    grad_tol = 1.0e-5,
    verbose = true)

	println()
    (verbose && obj isa StageCosts) && println("Differential Dynamic Programming")

    # allocate model data
    m_data = model_data(model, obj, w, h, T)
    m_data.x̄ .= x̄
    m_data.ū .= ū

    # allocate policy data
    p_data = policy_data(model, T)

    # allocate solver data
    s_data = solver_data(model, T)

    # compute objective
    s_data.obj = objective(m_data.obj, m_data.x̄, m_data.ū)

    for i = 1:max_iter
        # derivatives
        derivatives!(m_data)

        # backward pass
        backward_pass!(p_data, m_data)

        # forward pass
        forward_pass!(p_data, m_data, s_data)

        # check convergence
        grad_norm = norm(s_data.gradient, Inf)
        verbose && println("     iter: $i
             cost: $(s_data.obj)
             grad norm: $(grad_norm)")
        (!s_data.status || grad_norm < grad_tol) && break
    end

    return p_data, m_data, s_data
end

"""
    gradient of Lagrangian
        https://web.stanford.edu/class/ee363/lectures/lqr-lagrange.pdf
"""
function lagrangian_gradient!(s_data::SolverData, p_data::PolicyData, m_data::ModelData)
    T = m_data.T
    n = m_data.model.n
    m = m_data.model.m

    p = p_data.p
    Qx = p_data.Qx
    Qu = p_data.Qu

    for t = 1:T-1
        idx_x = (t - 1) * n .+ (1:n)
        s_data.gradient[idx_x] = Qx[t] - p[t]
        # NOTE: gradient wrt xT is satisfied implicitly

        idx_u = n * T + (t - 1) * m .+ (1:m)
        s_data.gradient[idx_u] = Qu[t]
    end
end

"""
    augmented Lagrangian solve
"""
function solve(model, obj::StageCosts, con_set::ConstraintSet, x̄, ū, w, h, T;
    max_iter = 10,
	max_al_iter = 5,
    grad_tol = 1.0e-5,
	con_tol = 1.0e-3,
	con_norm_type = Inf,
	ρ_init = 1.0,
	ρ_scale = 10.0,
    verbose = true)

	println()
	verbose && println("Differential Dynamic Programming")

	c_data = constraints_data(model, [c.p for c in con_set], T)
	cons = StageConstraints(con_set, c_data, T)

	obj_al = augmented_lagrangian(obj, cons, ρ = ρ_init)

	for i = 1:max_al_iter
		verbose && println("  al iter: $i")
		# primal minimization
		p_data, m_data, s_data = solve(model, obj_al, x̄, ū, w, h, T;
		    max_iter = max_iter,
		    grad_tol = grad_tol,
		    verbose = verbose)

		# update trajectories
		x̄ = m_data.x̄
		ū = m_data.ū
		objective(obj_al, x̄, ū)

		# constraint violation
		c_max = constraint_violation(obj_al.cons, x̄, ū,
			norm_type = con_norm_type)
		verbose && println("    c_max: $c_max\n")
		c_max <= con_tol && break

		# dual ascent
		augmented_lagrangian_update!(obj_al, s = ρ_scale)
	end

	return x̄, ū, obj_al
end
