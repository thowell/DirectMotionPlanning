# Model
include_model("quadruped")

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)

# Horizon
T = 5

# Time step
h = 0.1
tf = h * (T - 1)

function initial_configuration(model::Quadruped, θ1, θ2, θ3)
    q1 = zeros(model.nq)
    q1[3] = pi / 2.0
    q1[4] = -θ1
    q1[5] = θ2

    q1[8] = -θ1
    q1[9] = θ2

    q1[2] = model.l_thigh1 * cos(q1[4]) + model.l_calf1 * cos(q1[5])

    q1[10] = -θ3
    q1[11] = acos((q1[2] - model.l_thigh2 * cos(q1[10])) / model.l_calf2)

    q1[6] = -θ3
    q1[7] = acos((q1[2] - model.l_thigh2 * cos(q1[6])) / model.l_calf2)

    return q1
end

θ1 = pi / 4.0
θ2 = pi / 4.0
θ3 = pi / 3.0

q1 = initial_configuration(model, θ1, θ2, θ3)
visualize!(vis, model, [q1])


# control
# u1 = initial_torque(model, q1, h)[model.idx_u]
# ul <= u <= uu
_uu = Inf * ones(model.m)
_uu[model.idx_u] .= 0.0
_ul = zeros(model.m)
_ul[model.idx_u] .= -0.0
ul, uu = control_bounds(model, T, _ul, _uu)

xl, xu = state_bounds(model, T,
    x1 = [q1; q1])

# Objective
include_objective(["velocity", "nonlinear_stage"])
q_ref = linear_interpolation(q1, q1, T)
render(vis)
visualize!(vis, model, q_ref)
x0 = configuration_to_state(q_ref)

# penalty on slack variable
obj_penalty = PenaltyObjective(1.0e4, model.m)

# quadratic tracking objective
# Σ (x - xref)' Q (x - x_ref) + (u - u_ref)' R (u - u_ref)
obj_control = quadratic_time_tracking_objective(
    [1.0 * Diagonal(1.0e-5 * ones(model.n)) for t = 1:T],
    [1.0 * Diagonal([1.0e-5 * ones(model.nu)..., zeros(model.m - model.nu)...]) for t = 1:T-1],
    [[q1; q1] for t = 1:T],
    [zeros(model.m) for t = 1:T],
    1.0)

obj = MultiObjective([obj_penalty, obj_control])
# Constraints
include_constraints(["stage", "contact", "free_time", "loop"])

con_contact = contact_constraints(model, T)
con = multiple_constraints([con_contact])#,
    # con_free_time])#, con_loop,
	# con_pinned1, con_pinned2])

# cc = zeros(4)
# xx = rand(model.n)
# uu = rand(model.m)

# pinned2!(cc, xx, uu)

# Problem
prob = trajectory_optimization_problem(model,
               obj,
               T,
               xl = xl,
               xu = xu,
               ul = ul,
               uu = uu,
			   h = h,
               con = con)

# trajectory initialization
u0 = [1.0e-2 * rand(model.m) for t = 1:T-1] # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob)
# z0 .+= 0.001 * randn(prob.num_var)

# Solve
include_snopt()
# @load joinpath(@__DIR__, "quadruped_gait.jld2") z̄ q̄ ū τ̄ λ̄ b̄ h̄
@time z̄, info = solve(prob, copy(z0),
    nlp = :SNOPT7,
    tol = 1.0e-3, c_tol = 1.0e-3,
	max_iter = 5000,
    time_limit = 60 * 2, mapl = 5)

check_slack(z̄, prob)
x̄, ū = unpack(z̄, prob)
q̄ = state_to_configuration(x̄)
τ̄ = [u[model.idx_u] for u in ū]
λ̄ = [u[model.idx_λ] for u in ū]
b̄ = [u[model.idx_b] for u in ū]

[norm(fd(model, x̄[t+1], x̄[t], ū[t], zeros(model.d), h, t)) for t = 1:T-1]
# Visualize
vis = Visualizer()
render(vis)
# open(vis)
visualize!(vis, model, state_to_configuration(x̄), Δt = h)
