function ddp_solve!(prob::ProblemData;
    max_iter = 10,
    grad_tol = 1.0e-5,
    verbose = true)

	println()
    (verbose && prob.m_data.obj isa StageCosts) && println("Differential Dynamic Programming")

	# data
	p_data = prob.p_data
	m_data = prob.m_data
	s_data = prob.s_data

    # compute objective
    s_data.obj = objective(m_data, mode = :nominal)

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
end

"""
    gradient of Lagrangian
        https://web.stanford.edu/class/ee363/lectures/lqr-lagrange.pdf
"""
function lagrangian_gradient!(s_data::SolverData, p_data::PolicyData, n, m, T)
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
function constrained_ddp_solve!(prob::ProblemData;
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

	# initial penalty
	prob.m_data.obj.ρ = ρ_init

	for i = 1:max_al_iter
		verbose && println("  al iter: $i")

		# primal minimization
		ddp_solve!(prob,
		    max_iter = max_iter,
		    grad_tol = grad_tol,
		    verbose = verbose)

		# update trajectories
		objective(prob.m_data, mode = :nominal)

		# constraint violation
		c_max = constraint_violation(prob.m_data.obj.cons,
			prob.m_data.x̄, prob.m_data.ū,
			norm_type = con_norm_type)
		verbose && println("    c_max: $c_max\n")
		c_max <= con_tol && break

		# dual ascent
		augmented_lagrangian_update!(prob.m_data.obj, s = ρ_scale)
	end
end
