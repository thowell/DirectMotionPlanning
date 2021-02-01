const MOI = MathOptInterface

struct MOIGeneralProblem <: MOI.AbstractNLPEvaluator
    num_var::Int                 # number of decision variables
    num_con::Int                 # number of constraints
    primal_bounds
    constraint_bounds
end

struct MOIProblem <: MOI.AbstractNLPEvaluator
    num_var::Int                 # number of decision variables
    num_con::Int                 # number of constraints
    primal_bounds
    constraint_bounds
    prob
end

function moi_problem(prob)
    return MOIProblem(
        prob.num_var,
        prob.num_con,
        primal_bounds(prob),
        constraint_bounds(prob),
        prob)
end

pack(X, U, prob::MOIProblem) = pack(X, U, prob.prob)
unpack(Z, prob::MOIProblem) = unpack(Z, prob.prob)

primal_bounds(prob::MOI.AbstractNLPEvaluator) = prob.primal_bounds

constraint_bounds(prob::MOI.AbstractNLPEvaluator) = prob.constraint_bounds

function MOI.eval_objective(prob::MOIProblem, x)
    return eval_objective(prob.prob, x)
end

function MOI.eval_objective_gradient(prob::MOIProblem, grad_f, x)
    eval_objective_gradient!(grad_f, x, prob.prob)
end

function MOI.eval_constraint(prob::MOIProblem, g, x)
    eval_constraint!(g, x, prob.prob)
    return nothing
end

function MOI.eval_constraint_jacobian(prob::MOIProblem, jac, x)
    eval_constraint_jacobian!(jac, x, prob.prob)
    return nothing
end

function sparsity_jacobian(prob::MOIProblem)
    sparsity_jacobian(prob.prob)
end

MOI.features_available(prob::MOI.AbstractNLPEvaluator) = [:Grad, :Jac]
MOI.initialize(prob::MOI.AbstractNLPEvaluator, features) = nothing
MOI.jacobian_structure(prob::MOI.AbstractNLPEvaluator) = sparsity_jacobian(prob)
MOI.hessian_lagrangian_structure(prob::MOI.AbstractNLPEvaluator) = []
MOI.eval_hessian_lagrangian(prob::MOI.AbstractNLPEvaluator, H, x, σ, μ) = nothing

function solve(prob::MOI.AbstractNLPEvaluator, x0;
        tol = 1.0e-3,
        c_tol = 1.0e-2,
        max_iter = 1000,
        nlp = :ipopt,
        time_limit = 120,
        mipl = 0,
        mapl = 1)

    x_l, x_u = primal_bounds(prob)
    c_l, c_u = constraint_bounds(prob)

    nlp_bounds = MOI.NLPBoundsPair.(c_l, c_u)
    block_data = MOI.NLPBlockData(nlp_bounds, prob, true)

    if nlp == :ipopt
        solver = Ipopt.Optimizer()
        solver.options["max_iter"] = max_iter
        solver.options["tol"] = tol
        solver.options["constr_viol_tol"] = c_tol
        # solver.options["print_level"] = mapl
        # solver.options["linear_solver"] = "ma57"
    elseif nlp == :SNOPT7
        # solver = SNOPT7.Optimizer(
        #                           Major_feasibility_tolerance = c_tol,
        #                           Minor_feasibility_tolerance = tol,
        #                           Major_optimality_tolerance = tol,
        #                           Time_limit = time_limit,
        #                           Major_print_level = mapl,
        #                           Minor_print_level = mipl)

    solver = SNOPT7.Optimizer(MOI.MIN_SENSE,
        nothing, [], [], nothing,
        0, 0, 0,
        nothing,
        # Dict("Major_feasibility_tolerance" => c_tol,
        #      "Minor_feasibility_tolerance" => c_tol,
        #      "Minor_optimality_tolerance" => tol,
        #      "Major_optimality_tolerance" => tol,
        #      "Time_limit" => time_limit,
        #      "Major_print_level" => mapl,
        #      "Minor_print_level" => mipl))
        Dict("Major_feasibility_tolerance" => c_tol,
             "Major_optimality_tolerance" => tol,
             # "Minor_feasibility_tolerance" => c_tol,
             # "Minor_optimality_tolerance" => tol,
             "Time_limit" => time_limit,
             "Major_print_level" => mapl,
             "Minor_print_level" => mipl,
             "Major_iterations_limit" => 10000,
             "Minor_iterations_limit" => 200000,
             "Iterations_limit" => 1000000))
    else
        @error "nlp not setup"
    end

    x = MOI.add_variables(solver, prob.num_var)

    for i = 1:prob.num_var
        xi = MOI.SingleVariable(x[i])
        MOI.add_constraint(solver, xi, MOI.LessThan(x_u[i]))
        MOI.add_constraint(solver, xi, MOI.GreaterThan(x_l[i]))
        MOI.set(solver, MOI.VariablePrimalStart(), x[i], x0[i])
    end

    # Solve the problem
    MOI.set(solver, MOI.NLPBlock(), block_data)
    MOI.set(solver, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(solver)

    # Solver info
    info = [] #TODO add info from solve

    # Get the solution
    return MOI.get(solver, MOI.VariablePrimal(), x), info
end
