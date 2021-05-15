function ddp_solve!(prob::ProblemData;
    max_iter = 10,
    grad_tol = 1.0e-5,
    verbose = true,
	cache = false)

	println()
    (verbose && prob.m_data.obj isa StageCosts) && printstyled("Differential Dynamic Programming\n",
		color = :red, bold = true)

	# data
	p_data = prob.p_data
	m_data = prob.m_data
	s_data = prob.s_data

    # compute objective
    # s_data.obj = objective(m_data, mode = :nominal)
	objective!(s_data, m_data, mode = :nominal)
	# println("obj")
    for i = 1:max_iter
        # derivatives
        derivatives!(m_data)
		# println("derivatives")

		# backward pass
        backward_pass!(p_data, m_data)
		# println("backward pass")

        # forward pass
        forward_pass!(p_data, m_data, s_data)
		# println("forward pass")

		# cache solver data
		cache && cache!(s_data)

        # check convergence
        grad_norm = norm(s_data.gradient, Inf)
        verbose && println("     iter: $i
             cost: $(s_data.obj)
			 grad_norm: $(grad_norm)
			 c_max: $(s_data.c_max)
			 α: $(s_data.α)")
		grad_norm < grad_tol && break
        !s_data.status && break
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
        idx_x = (t == 1 ? 0 : (t - 1) * n[t-1]) .+ (1:n[t])
        s_data.gradient[idx_x] = Qx[t] - p[t]
        # NOTE: gradient wrt xT is satisfied implicitly

        idx_u = sum(n) + (t == 1 ? 0 : (t - 1) * m[t-1]) .+ (1:m[t])
        s_data.gradient[idx_u] = Qu[t]
    end
end

function lagrangian_gradient!(s_data::SolverData, p_data::PolicyData, m_data::ModelData)
	lagrangian_gradient!(s_data, p_data,
		m_data.n, m_data.m, m_data.T)
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
	ρ_max = 1.0e8,
	cache = false,
    verbose = true)

	println()
	verbose && printstyled("Differential Dynamic Programming\n",
		color = :red, bold = true)

	# initial penalty
	for (t, ρ) in enumerate(prob.m_data.obj.ρ)
		prob.m_data.obj.ρ[t] = ρ_init .* ρ
	end

	for i = 1:max_al_iter
		verbose && println("  al iter: $i")

		# primal minimization
		ddp_solve!(prob,
		    max_iter = max_iter,
		    grad_tol = grad_tol,
			cache = cache,
		    verbose = verbose)

		# update trajectories
		objective!(prob.s_data, prob.m_data, mode = :nominal)

		# constraint violation
		# c_max = constraint_violation(prob.m_data.obj.cons,
		# 	prob.m_data.x̄, prob.m_data.ū,
		# 	norm_type = con_norm_type)
		# verbose && println("    c_max: $(prob.s_data.c_max)\n")
		prob.s_data.c_max <= con_tol && break

		# dual ascent
		augmented_lagrangian_update!(prob.m_data.obj,
			s = ρ_scale, max_penalty = ρ_max)
	end
end
