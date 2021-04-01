# Model
include_model("particle")
model = Particle{Discrete, FixedTime}(n, m, d,
				 mass, 0.0, μ,
				 nq,
				 nu,
				 nc,
				 nf,
				 nb,
				 ns,
				 idx_u,
				 idx_λ,
				 idx_b,
				 idx_ψ,
				 idx_η,
				 idx_s)
# Horizon
T = 51

# Time step
tf = 1.0
h = tf / (T-1)

# Path
px = range(0.0, stop = tf, length = T+1)
py = 0.5 * sin.(2 * π * px)

plot(px, py, aspect_ratio = :equal)
q_ref = [[px[t]; py[t]; 1.0] for t = 1:length(px)]
x_ref = configuration_to_state(q_ref)

# Bounds
_uu = Inf * ones(model.m)
_ul = zeros(model.m)
_ul[model.idx_u] .= -Inf

ul, uu = control_bounds(model, T, _ul, _uu)

# Initial and final states
q1 = q_ref[1]#[0.0, 0.0, 1.0]
# v1 = [3.0, 5.0, 0.0]
# v2 = v1 - C_func(model, q1, v1) * h
# q2 = q1 + 0.5 * h * (v1 + v2)
#
# x1 = [q1; q2]

xl, xu = state_bounds(model, T, x1 = [q_ref[1]; q_ref[2]], xT = [q_ref[end-1]; q_ref[end]])

# Objective
obj_tracking = quadratic_tracking_objective(
    [Diagonal(1000.0 * ones(model.n)) for t = 1:T],
    [Diagonal([1.0e-3, 1.0e-3, 1.0e-3,
		ones(model.nc)..., ones(model.nb)...,
		zeros(model.m - model.nu - model.nc - model.nb)...])
		for t = 1:T-1],
   	x_ref,
    [zeros(model.m) for t = 1:T])
obj_penalty = PenaltyObjective(1.0e5, model.m)

obj = MultiObjective([obj_tracking, obj_penalty])

# Constraints
include_constraints(["contact", "loop"])
con_contact = contact_constraints(model, T)
# con_loop = loop_constraints(model, collect([(2:model.nq)..., (nq .+ (2:model.nq))...]), 1, T)
con = multiple_constraints([con_contact])#, con_loop])

# Problem
prob = trajectory_optimization_problem(model,
               obj,
               T,
               h = h,
               xl = xl,
               xu = xu,
               ul = ul,
               uu = uu,
               con = con)

# Trajectory initialization
x0 = deepcopy(x_ref) #linear_interpolation(x1, x1, T) # linear interpolation on state
u0 = [0.001 * rand(model.m) for t = 1:T-1] # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem
@time z̄, info = solve(prob, copy(z0), tol = 1.0e-5, c_tol = 1.0e-5)

check_slack(z̄, prob)
x̄, ū = unpack(z̄, prob)
q̄ = state_to_configuration(x̄)

q = state_to_configuration(x̄)
u = [u[model.idx_u] for u in ū]
γ = [u[model.idx_λ] for u in ū]
b = [u[model.idx_b] for u in ū]
h̄ = mean(h̄)
@save joinpath(@__DIR__, "particle_sinusoidal.jld2") z̄ x̄ ū h̄ q u γ b

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
visualize!(vis, model,
    q̄,
    Δt = h)

plot(hcat(q_ref...)', color = :black, width = 2.0)
plot!(hcat(q̄...)', color = :red, width = 1.0)

vT = (q̄[end] - q̄[end-1]) / h
v1 = (q̄[2] - q̄[1]) / h
@show norm(v1 - vT)
# open(vis)
#
# vT = (q_ref[end] - q_ref[end-1]) / h
# v1 = (q_ref[2] - q_ref[1]) / h
# @show norm(v1 - vT)
