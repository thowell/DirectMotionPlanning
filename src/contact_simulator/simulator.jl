include(joinpath(pwd(), "src/constraints/contact.jl"))
# include(joinpath(pwd(), "src/constraints/contact_no_slip.jl"))

# Simulate contact model one step
function step_contact(model::Model{<: Integration, FixedTime}, x1, u1, w1, h;
        tol_c = 1.0e-5, tol_opt = 1.0e-5, tol_s = 1.0e-4, nlp = :ipopt)
    # Horizon
    T = 2

    # Objective
    obj = PenaltyObjective(1.0e5, model.m)

    # Constraints
    con_dynamics = dynamics_constraints(model, T; w = [w1])
    con_contact = contact_constraints(model, T)
    con = multiple_constraints([con_dynamics, con_contact])

    # Bounds
    _uu = Inf * ones(model.m)
    _uu[model.idx_u] .= u1
    _ul = zeros(model.m)
    _ul[model.idx_u] .= u1
    ul, uu = control_bounds(model, T, _ul, _uu)

    xl, xu = state_bounds(model, T, x1 = x1)

    # Problem
    prob = trajectory_optimization_problem(model,
                   obj,
                   T,
                   h = h,
                   xl = xl,
                   xu = xu,
                   ul = ul,
                   uu = uu,
                   con = con,
                   dynamics = false)

    # Trajectory initialization
    x0 = [x1 for t = 1:T]
    u0 = [[u1; 1.0e-5 * rand(model.m - model.nu)] for t = 1:T-1] # random controls

    # Pack trajectories into vector
    z0 = pack(x0, u0, prob)

    @time z, info = solve(prob, copy(z0),
        tol = tol_opt, c_tol = tol_c, mapl = 0, nlp = nlp)

    @assert check_slack(z, prob) < tol_s
    x, u = unpack(z, prob)

    return x[end]
end
