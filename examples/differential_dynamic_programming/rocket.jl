using Plots
include_ddp()

# Model
include_model("rocket3D")

function fd(model::Rocket3D{Midpoint, FixedTime}, x, u, w, h, t)
	return view(x, 1:model.n) + h * f(model, view(x, 1:model.n) + 0.5 * h * f(model, view(x, 1:model.n), view(u, 1:model.m), w), view(u, 1:model.m), w)
end

n = model.n
m = model.m

# Time
T = 201
h = 0.01

# Initial conditions, controls, disturbances
x1 = zeros(model.n)
x1[1] = 1.0
x1[2] = 1.0
x1[3] = 10.0
mrp = MRP(RotY(-0.5 * π) * RotX(0.0 * π))
x1[4:6] = [mrp.x; mrp.y; mrp.z]

# visualize!(vis, model, [x1], Δt = h)

xT = zeros(model.n)
# xT[1] = 2.5
# xT[2] = 0.0
xT[3] = model.length

u_ref = [0.0; 0.0; 0.0]#model.mass * 9.81]
ū = [u_ref + [1.0e-2; 1.0e-2; 1.0e-2] .* randn(model.m) for t = 1:T-1]
w = [zeros(model.d) for t = 1:T-1]

# Rollout
x̄ = rollout(model, x1, ū, w, h, T)
# x̄ = linear_interpolation(x1, xT, T)
# plot(hcat(x̄...)')

# Objective
Q = h * [(t < T ? 1.0 * Diagonal([1.0e-1 * ones(3); 0.0 * ones(3); 1.0e-1 * ones(3); 1000.0 * ones(3)])
        : 0.0 * Diagonal(0.0 * ones(model.n))) for t = 1:T]
q = h * [-2.0 * Q[t] * xT for t = 1:T]

R = h * [Diagonal([10000.0; 10000.0; 100.0]) for t = 1:T-1]
r = h * [-2.0 * R[t] * u_ref  for t = 1:T-1]

obj = StageCosts([QuadraticCost(Q[t], q[t],
	t < T ? R[t] : nothing, t < T ? r[t] : nothing) for t = 1:T], T)

function g(obj::StageCosts, x, u, t)
	T = obj.T
    if t < T
		Q = obj.cost[t].Q
		q = obj.cost[t].q
	    R = obj.cost[t].R
		r = obj.cost[t].r
        return x' * Q * x + q' * x + u' * R * u + r' * u
    elseif t == T
		Q = obj.cost[T].Q
		q = obj.cost[T].q
        return x' * Q * x + q' * x
    else
        return 0.0
    end
end

# Constraints
p = [t < T ? 2 * m : n for t = 1:T]
info_t = Dict(:ul => [-5.0; -5.0; 0.0], :uu => [5.0; 5.0; 100.0], :inequality => (1:2 * m))
info_T = Dict(:xT => xT)
con_set = [StageConstraint(p[t], t < T ? info_t : info_T) for t = 1:T]

function c!(c, cons::StageConstraints, x, u, t)
	T = cons.T
	p = cons.con[t].p

	if t < T
		ul = cons.con[t].info[:ul]
		uu = cons.con[t].info[:uu]
		c .= [ul - u; u - uu]
	elseif t == T
		xT = cons.con[T].info[:xT]
		c .= x - xT
	else
		nothing
	end
end

prob = problem_data(model, obj, con_set, copy(x̄), copy(ū), w, h, T)

# Solve
@time constrained_ddp_solve!(prob,
    max_iter = 1000, max_al_iter = 10,
	con_tol = 1.0e-3,
	ρ_init = 1.0, ρ_scale = 10.0)

x, u = current_trajectory(prob)
x̄, ū = nominal_trajectory(prob)

# Trajectories
plot(hcat(ū...)', linetype = :steppost)
plot(hcat(x̄...)[1:3, :]', linetype = :steppost)

# Visualize
include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
# open(vis)
visualize!(vis, model, x̄, Δt = h)

ū_fixed_time = ū

# Visualize
obj_rocket = joinpath(pwd(), "models/starship/Starship.obj")
mtl_rocket = joinpath(pwd(), "models/starship/Starship.mtl")
ctm = ModifiedMeshFileObject(obj_rocket, mtl_rocket, scale=1.0)
setobject!(vis["rocket"]["starship"], ctm)

settransform!(vis["rocket"]["starship"],
	compose(Translation(0.0, 0.0, -model.length),
		LinearMap(0.25 * RotY(0.0) * RotZ(0.5 * π) * RotX(0.5 * π))))

default_background!(vis)
settransform!(vis["rocket"],
	compose(Translation(0.0, 0.0, 0.0),
	LinearMap(RotY(0.0))))
