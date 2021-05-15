using Plots
include_ddp()

# Model
include_model("rocket3D")

function fd(model::Rocket3D{Midpoint, FixedTime}, x, u, w, h, t)
	if t == 1
		_h = h + u[4]
		# _hh = u[4]
	else
		_h = x[13]
	end
	# h = 1.0 / (1.0 + exp(-h))
	return [view(x, 1:model.n) + _h * f(model, view(x, 1:model.n) + 0.5 * _h * f(model, view(x, 1:model.n), view(u, 1:model.m), w), view(u, 1:model.m), w);
	        _h]
end

n = [t == 1 ? model.n : model.n + 1 for t = 1:T]
m = [t == 1 ? model.m + 1 : model.m for t = 1:T]

fd(model, rand(model.n), rand(model.m + 1), zeros(model.d), 1.0, 1)
fd(model, rand(model.n + 1), rand(model.m), zeros(model.d), 1.0, 2)

# Time
T = 201
h = 0.01

# Initial conditions, controls, disturbances
x1 = zeros(model.n)
x1[1] = 1.0
x1[2] = 1.0
x1[3] = 10.0
mrp = MRP(RotY(-0.5 * π) * RotX(0.25 * π))
x1[4:6] = [mrp.x; mrp.y; mrp.z]

xT = zeros(model.n)
# xT[1] = 2.5
# xT[2] = 0.0
xT[3] = model.length

u_ref = [0.0; 0.0; 0.0]
ū = [t == 1 ? [ū_fixed_time[t]; 0.0] : ū_fixed_time[t] for t = 1:T-1]
# ū = [t == 1 ? [[1.0e-2; 1.0e-2; 1.0e-2] .* randn(model.m); 0.0] : [1.0e-2; 1.0e-2; 1.0e-2] .* randn(model.m) for t = 1:T-1]
w = [zeros(model.d) for t = 1:T-1]

# Rollout
x̄ = rollout(model, x1, ū, w, h, T)

# # Objective
# Q = [(t == 1 ? Diagonal([1.0e-3 * ones(3); 0.0 * ones(3); 1.0e-3 * ones(3); 10.0 * ones(3)])
#         : (t == T ? Diagonal([0.0 * ones(model.n); 0.0])
# 		: Diagonal([1.0e-3 * ones(3); 0.0 * ones(3); 1.0e-3 * ones(3); 10.0 * ones(3); 0.0]))) for t = 1:T]
# q = [(t == 1 ? -2.0 * Q[t] * xT
#  	 : -2.0 * Q[t] * [xT; 0.0]) for t = 1:T]
#
# R = [(t == 1 ? Diagonal([100.0; 100.0; 1.0; 1.0e-6])
# 	 : Diagonal([100.0; 100.0; 1.0])) for t = 1:T-1]
# r = [(t == 1 ? [-2.0 * R[t][1:model.m, 1:model.m] * u_ref; -2.0 * 1.0 * 0.0]
# 	 : -2.0 * R[t] * u_ref)  for t = 1:T-1]
#
# obj = StageCosts([QuadraticCost(Q[t], q[t],
# 	t < T ? R[t] : nothing, t < T ? r[t] : nothing) for t = 1:T], T)
#
# function g(obj::StageCosts, x, u, t)
# 	T = obj.T
#     if t < T
# 		Q = obj.cost[t].Q
# 		q = obj.cost[t].q
# 	    R = obj.cost[t].R
# 		r = obj.cost[t].r
#         return x' * Q * x + q' * x + u' * R * u + r' * u
#     elseif t == T
# 		Q = obj.cost[T].Q
# 		q = obj.cost[T].q
#         return x' * Q * x + q' * x
#     else
#         return 0.0
#     end
# end

# Objective
Q = [(t < T ? 1.0 * Diagonal([1.0e-1 * ones(3); 0.0 * ones(3); 1.0e-1 * ones(3); 1000.0 * ones(3)])
        : 0.0 * Diagonal(0.0 * ones(model.n))) for t = 1:T]
q = [-2.0 * Q[t] * xT for t = 1:T]

R = [Diagonal([10000.0; 10000.0; 100.0]) for t = 1:T-1]
r = [-2.0 * R[t] * u_ref  for t = 1:T-1]

obj = StageCosts([NonlinearCost() for t = 1:T], T)

function g(obj::StageCosts, x, u, t)
	T = obj.T
    if t < T
		t == 1 ? (_h = h + u[end]) : (_h = x[end])
		view(u, 1:model.m)
        return _h * (view(x, 1:model.n)' * Q[t] * view(x, 1:model.n) + q[t]' * view(x, 1:model.n) + view(u, 1:model.m)' * R[t] * view(u, 1:model.m) + r[t]' * view(u, 1:model.m))
		_h = x[end]
        return _h * (view(x, 1:model.n)' * Q[t] * view(x, 1:model.n) + q[t]' * view(x, 1:model.n))
    else
        return 0.0
    end
end

# g(obj, x̄[1], ū[1], 1)
# g(obj, x̄[2], ū[2], 2)
# g(obj, x̄[T], nothing, T)


# Constraints
p = [t < T ? (t == 1 ? 2 * model.m + 2 : 2 * model.m) : model.n for t = 1:T]
info_1 = Dict(:ul => [-5.0; -5.0; 0.0; 0.0], :uu => [5.0; 5.0; 100.0; 0.0], :inequality => (1:(2 * model.m + 2)))
info_t = Dict(:ul => [-5.0; -5.0; 0.0], :uu => [5.0; 5.0; 100.0], :inequality => (1:2 * model.m))
info_T = Dict(:xT => xT)
con_set = [StageConstraint(p[t], t < T ? (t == 1 ? info_1 : info_t) : info_T) for t = 1:T]


function c!(c, cons::StageConstraints, x, u, t)
	T = cons.T
	p = cons.con[t].p

	if t < T
		ul = cons.con[t].info[:ul]
		uu = cons.con[t].info[:uu]
		c .= [ul - u; u - uu]
	elseif t == T
		xT = cons.con[T].info[:xT]
		c .= view(x, 1:model.n) - xT
	else
		nothing
	end
end

prob = problem_data(model, obj, con_set, copy(x̄), copy(ū), w, h, T,
	n = n, m = m)

prob.m_data.obj.ρ[1][4] *= 1000.0
prob.m_data.obj.ρ[1][4]

# Solve
@time constrained_ddp_solve!(prob,
    max_iter = 1000, max_al_iter = 10,
	con_tol = 1.0e-3,
	ρ_init = 1.0, ρ_scale = 10.0)

x, u = current_trajectory(prob)
x̄, ū = nominal_trajectory(prob)

# Time step
@show u[1][end]
x[2][end]

# Trajectories
plot(hcat([ut[1:model.m] for ut in u]...)', linetype = :steppost)
plot(hcat([xt[1:model.n] for xt in x]...)[1:3, :]', linetype = :steppost)

# Visualize
include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
# open(vis)
visualize!(vis, model, x̄, Δt = ū[1][4])
