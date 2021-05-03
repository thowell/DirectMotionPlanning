using Random, Plots
include_ddp()

# Model
include_model("satellite")

# Time
T = 40
h = 0.1
tf = h * (T - 1)

# Reference trajectory
# https://en.wikipedia.org/wiki/Viviani%27s_curve
# https://mathworld.wolfram.com/VivianisCurve.html
t = range(-2.0 * π, stop = 2.0 * π, length = T)
a = 0.05
xf = a * (1.0 .+ cos.(t))
yf = a * sin.(t)
zf = 2.0 * a * sin.(0.5 .* t)

plot(xf, zf, aspect_ratio = :equal)
plot(xf, yf, aspect_ratio = :equal)
plot(yf, zf, aspect_ratio = :equal)
plot(xf, yf, zf, aspect_ratio = :equal)
pf = [RotZ(0.0 * π) * [xf[t]; yf[t]; zf[t]] for t = 1:T]

ref1 = [a * (1.0 .+ cos.(-π));
		a * sin.(-π);
		2.0 * a * sin.(0.5 .* -π)]
ref2 = [a * (1.0 .+ cos.(0.0));
		a * sin.(0.0);
		2.0 * a * sin.(0.5 .* 0.0)]
ref3 = [a * (1.0 .+ cos.(π));
		a * sin.(0.0);
		2.0 * a * sin.(0.5 .* π)]
ref4 = [a * (1.0 .+ cos.(2.0 * π));
		a * sin.(2.0 * π);
		2.0 * a * sin.(0.5 .* 2.0 * π)]

# Initial conditions, controls, disturbances
mrp1 = MRP(RotXYZ(0.0, 0.0, 0.0))
mrpT = MRP(RotXYZ(0.0, 0.0, 0.0))

x1 = [mrp1.x, mrp1.y, mrp1.z, 0.0, 0.0, 0.0]
xT = [mrpT.x, mrpT.y, mrpT.z, 0.0, 0.0, 0.0]

ū = [1.0e-3 * rand(model.m) for t = 1:T-1]
w = [zeros(model.d) for t = 1:T-1]

# Rollout
x̄ = rollout(model, x1, ū, w, h, T)


obj = StageCosts([NonlinearCost() for t = 1:T], T)

function g(obj::StageCosts, x, u, t)
	T = obj.T
	ω = view(x, 4:6)

	J = 0.0

	# tracking
	p = kinematics(model, x)
	J += transpose(p - pf[t]) * Diagonal(1000.0 * [1.0; 1.0; 1.0]) * (p - pf[t])

	# control limits
	if t < T
		# energy
		J += 0.5 * transpose(ω) * model.J * ω
		J += 1.0e-2 * transpose(u[1:3]) * u[1:3]
	end

	return J / T
end


# Constraints
p_con = [t == T ? model.n : ((t == 10 || t == 20 || t == 30) ? 3 : 0) for t = 1:T]

info_t = Dict(:x1 => ref1, :x2 => ref2, :x3 => ref3)
info_T = Dict(:xT => xT)

con_set = [StageConstraint(p_con[t], t < T ? info_t : info_T) for t = 1:T]

function c!(c, cons::StageConstraints, x, u, t)
	T = cons.T
	n = model.n

	if t == 10
		c[1:3] .= kinematics(model, x) - cons.con[t].info[:x1]
	end

	if t == 20
		c[1:3] .= kinematics(model, x) - cons.con[t].info[:x2]
	end

	if t == 30
		c[1:3] .= kinematics(model, x) - cons.con[t].info[:x3]
	end

	# if t == 40
	# 	c[1:3] .= kinematics(model, x) - cons.con[t].info[:x4]
	# end

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
px = [pt[1] for pt in p]
py = [pt[2] for pt in p]
pz = [pt[3] for pt in p]

plot(yf, zf)
plot!([pt[2] for pt in p], [pt[3] for pt in p])

plot(xf, zf)
plot(xf, yf)
plot(xf)
plot(yf)
plot(zf)

plot(xf, yf, zf, aspect_ratio = :equal)
plot!(px, py, pz, aspect_ratio = :equal)
