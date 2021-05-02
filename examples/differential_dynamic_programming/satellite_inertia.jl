using Random, Plots
include_ddp()

# Model
include_model("satellite_inertia")

# Time
T = 26
h = 0.1
tf = h * (T - 1)

# Reference trajectory
tf = range(0.0, stop = 2.0 * pi, length = T)
a = 0.05
yf = a * sin.(tf)
zf = a * sin.(tf) .* cos.(tf)

pf = [RotZ(0.0 * π) * [0.1; yf[t]; zf[t]] for t = 1:T]

plot(yf, zf)

# Initial conditions, controls, disturbances
mrp1 = MRP(RotXYZ(0.0, 0.0, 0.0))
mrpT = MRP(RotXYZ(0.0, 0.0, 0.0))

x1 = [mrp1.x, mrp1.y, mrp1.z, 0.0, 0.0, 0.0]
xT = [mrpT.x, mrpT.y, mrpT.z, 0.0, 0.0, 0.0]

ū = [[1.0e-1 * rand(3); 1.0; 2.0; 3.0] for t = 1:T-1]
w = [zeros(model.d) for t = 1:T-1]

# Rollout
x̄ = rollout(model, x1, ū, w, h, T)

obj = StageCosts([NonlinearCost() for t = 1:T], T)

function g(obj::StageCosts, x, u, t)
	T = obj.T
	ω = view(x, 4:6)

	#
	J = 0.0

	# tracking
	p = kinematics(model, x)
	J += transpose(p - pf[t]) * Diagonal(100000.0 * [1.0; 1.0; 1.0]) * (p - pf[t])

	# control limits
	if t < T
		# energy
		θ = Diagonal(view(u, 4:6))
		J += 0.5 * transpose(ω) * θ * ω

		J += 1.0e-5 * transpose(u[1:3]) * u[1:3]
	end

	return J
end

# Constraints
p_con = [t == T ? model.n : 2 * model.m for t = 1:T]

info_t = Dict(:JL => [0.5; 1.0; 1.5], :JU => [2.0; 4.0; 6.0])
info_T = Dict(:xT => xT)

con_set = [StageConstraint(p_con[t], t < T ? info_t : info_T) for t = 1:T]

function c!(c, cons::StageConstraints, x, u, t)
	T = cons.T
	n = model.n

	if t < T
		c[1:3] .= cons.con[t].info[:JL] - u[4:6]
		c[4:6] .= u[4:6] - cons.con[t].info[:JU]
	end

	if t == T
		c[1:n] .= x - cons.con[T].info[:xT]
	end
end

prob = problem_data(model, obj, con_set, copy(x̄), copy(ū), w, h, T)

# Solve
@time constrained_ddp_solve!(prob,
    max_iter = 1000, max_al_iter = 10,
	ρ_init = 1.0, ρ_scale = 10.0,
	con_tol = 1.0e-3,
	cache = false)

x, u = current_trajectory(prob)
x̄, ū = nominal_trajectory(prob)

# Visualize
include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
visualize!(vis, model, x, Δt = h)

p = [kinematics(model, view(xt, 1:3)) for xt in x]
plot(yf, zf)
plot!([pt[2] for pt in p], [pt[3] for pt in p])
