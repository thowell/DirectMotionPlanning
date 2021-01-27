# Modified dynamics methods (for systems with a single contact) for simulation using v, q

function dynamics_sim(model::Model{Discrete, FixedTime}, v1, q1, q2, q3, u, λ, b, w, h)
   (M_func(model, q1) * v1
   - M_func(model, q2) * (SVector{model.nq}(q3) - SVector{model.nq}(q2)) / h
   + h * (w
   + transpose(B_func(model, q3)) * SVector{model.nu}(u)
   + transpose(N_func(model, q3)) * SVector{model.nc}(λ)
   + transpose(P_func(model, q3)) * SVector{model.nb}(b)
   - G_func(model, q2)))
end

function maximum_dissipation_sim(model::Model{Discrete, FixedTime}, q2, q3, ψ, η, h)
   ψ_stack = ψ[1] * ones(model.nb)
   return P_func(model, q3) * (q3 - q2) / h + ψ_stack - η
end

function no_slip_sim(model::Model{Discrete, FixedTime}, q2, q3, λ, h)
   return (λ' * _P_func(model, q3) * (q3 - q2) / h)[1]
end

function friction_cone_sim(model::Model{Discrete, FixedTime}, λ, b)
   return @SVector [model.μ * λ[1] - sum(b)]
end

# Simulator
mutable struct MOISimulator <: MOI.AbstractNLPEvaluator
   num_var::Int                 # number of decision variables
   num_con::Int                 # number of constraints
   primal_bounds
   constraint_bounds

   model
   slack_penalty
   v1
   q1
   q2
   u
   w
   h
end

function simulator_problem(model, v1, q1, q2, u, w, h; slack_penalty = 1.0e5)
   num_var = model.nq + model.nc + model.nb + model.nc + model.nb + model.ns
   num_con = model.nq + model.nc + model.nc + model.nb +  3

   zl = zeros(num_var)
   zl[1:model.nq] .= -Inf
   zu = Inf * ones(num_var)

   cl = zeros(num_con)
   cu = zeros(num_con)
   cu[model.nq .+ (1:model.nc)] .= Inf
   cu[model.nq + model.nc .+ (1:model.nc)] .= Inf
   cu[model.nq + model.nc + model.nc + model.nb + 1] = Inf
   cu[model.nq + model.nc + model.nc + model.nb + 2] = Inf
   cu[model.nq + model.nc + model.nc + model.nb + 3] = Inf

   MOISimulator(num_var,
	   num_con,
	   (zl, zu),
	   (cl, cu),
	   model,
	   slack_penalty,
	   v1,
	   q1,
	   q2,
	   u,
	   w,
	   h)
end
# prob_sim = simulator_problem(model, ones(nq), ones(nq), ones(nq), zeros(nu), ones(nq), h)
# z0 = rand(prob_sim.num_var)

function MOI.eval_objective(prob::MOISimulator, x)
   return prob.slack_penalty * x[prob.num_var]
end

# MOI.eval_objective(prob_sim, z0)

function MOI.eval_objective_gradient(prob::MOISimulator, grad_f, x)
   grad_f .= 0.0
   grad_f[prob.num_var] = prob.slack_penalty
   return nothing
end

# ∇obj = zeros(prob_sim.num_var)
# MOI.eval_objective_gradient(prob_sim, ∇obj, z0)

function MOI.eval_constraint(prob::MOISimulator, g, x)
   model = prob.model
   q3 = view(x, 1:model.nq)
   λ = view(x, model.nq .+ (1:model.nc))
   b = view(x, model.nq + model.nc .+ (1:model.nb))
   ψ = view(x, model.nq + model.nc + model.nb .+ (1:model.nc))
   η = view(x, model.nq + model.nc + model.nb + model.nc .+ (1:model.nb))
   s = view(x, model.nq + model.nc + model.nb + model.nc + model.nb .+ (1:model.ns))

   g[1:model.nq] = dynamics_sim(model,
	   prob.v1, prob.q1, prob.q2, q3, prob.u, λ, b, prob.w, prob.h)
   g[model.nq .+ (1:model.nc)] = ϕ_func(model, q3)
   g[model.nq + model.nc .+ (1:model.nc)] = friction_cone_sim(model, λ, b)
   g[model.nq + model.nc + model.nc .+ (1:model.nb)] = maximum_dissipation_sim(model,
	   prob.q2, q3, ψ, η, prob.h)
   g[model.nq + model.nc + model.nc + model.nb + 1] = s[1] - (λ' * ϕ_func(model, q3))[1]
   g[model.nq + model.nc + model.nc + model.nb + 2] = s[1] - (ψ' * friction_cone_sim(model, λ, b))[1]
   g[model.nq + model.nc + model.nc + model.nb + 3] = s[1] - (η' * b)[1]

   return nothing
end
# prob_sim.h
# c0 = zeros(prob_sim.num_con)
# MOI.eval_constraint(prob_sim, c0, z0)

function MOI.eval_constraint_jacobian(prob::MOISimulator, jac, x)
   con!(g, z) = MOI.eval_constraint(prob, g, z)
   ForwardDiff.jacobian!(reshape(jac, prob.num_var, prob.num_con), con!, zeros(prob.num_con), x)
   return nothing
end

# ∇c0 = vec(zeros(prob_sim.num_con, prob_sim.num_var))
# MOI.eval_constraint_jacobian(prob_sim, ∇c0, z0)
# #
function sparsity_jacobian(prob::MOISimulator)
   row = []
   col = []

   r = (1:prob.num_con)
   c = (1:prob.num_var)

   row_col!(row, col, r, c)

   return collect(zip(row, col))
end

# sparsity_jacobian(prob_sim)
#
# @time solve(prob_sim, z0)

function step_contact(model, v1, q1, q2, u, w, h)
   prob = simulator_problem(model, v1, q1, q2, u, w, h)
   z0 = [copy(q2); 1.0e-5 * ones(model.nc + model.nb + model.nc + model.nb + model.ns)]
   @time z, info = solve(prob, copy(z0), tol = 1.0e-5, c_tol = 1.0e-5, mapl = 0, nlp = :SNOPT7)

   @assert z[end] < 1.0e-4

   return z[1:model.nq]
end
