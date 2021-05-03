using Random, Plots
include_ddp()

# Model
include_model("satellite")

function f(model::Satellite, z, u, w)
      # states
      r = view(z, 1:3)
      ω = view(z, 4:6)

      # controls
      τ = view(u, 1:3)
	  J = Diagonal(w[1:3])

      SVector{6}([0.25 * ((1.0 - r' * r) * ω - 2.0 * cross(ω, r) + 2.0 * (ω' * r) * r);
                  J \ (τ - cross(ω, J * ω))])
end

function fd(model::Satellite{Midpoint, FixedTime}, x, u, w, h, t)
	if t == 1
		J = view(u, 4:6)
	else
		J = view(x, 7:9)
	end

	return [view(x, 1:6) + h * f(model, view(x, 1:6) + 0.5 * h * f(model, view(x, 1:6), view(u, 1:3), J), view(u, 1:3), J);
		    J]
end

# Time
T = 40
h = 0.1
tf = h * (T - 1)

# Modified dynamics dimensions
n = [t == 1 ? model.n : model.n + 3 for t = 1:T]
m = [t == 1 ? model.m + 3 : model.m for t = 1:T]

J_nominal = [1.0; 2.0; 3.0]

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

ū = [t == 1 ? [1.0e-3 * randn(model.m); J_nominal] : 1.0e-3 * randn(model.m) for t = 1:T-1]
w = [zeros(0) for t = 1:T-1]

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
		if t == 1
			inertia = Diagonal(u[4:6])
			J += 1.0e-5 * transpose(u[4:6] - J_nominal) * (u[4:6] - J_nominal)
		else
			inertia = Diagonal(x[7:9])
		end
		J += 0.5 * transpose(ω) * inertia * ω
		J += 1.0e-2 * transpose(u[1:3]) * u[1:3]
	end

	return J / T
end


# Constraints
p_con = [t == T ? model.n : ((t == 10 || t == 20 || t == 30) ? 3 + 6 : (t == 1 ? 12 : 6)) for t = 1:T]
uL = -10.0 * ones(model.m)
uU = 10.0 * ones(model.m)
# info_1 = Dict(:uL => uL, :uU => uU, :JL => [0.9; 1.8; 2.7], :JU => [1.1; 2.2; 3.3], :inequality => (1:12))
info_1 = Dict(:uL => uL, :uU => uU, :JL => [0.9; 1.8; 2.7], :JU => [1.1; 2.2; 3.3], :inequality => (1:12))
info_p = Dict(:uL => uL, :uU => uU, :x1 => ref1, :x2 => ref2, :x3 => ref3, :inequality => (4:9))
info_t = Dict(:uL => uL, :uU => uU, :inequality => (1:6))

info_T = Dict(:xT => xT)

con_set = [StageConstraint(p_con[t], t < T ? (t == 1 ? info_1 : ((t == 10 || t == 20 || t == 30) ? info_p : info_t)) : info_T) for t = 1:T]

function c!(c, cons::StageConstraints, x, u, t)
	T = cons.T
	n = model.n

	if t == 1
		c[1:3] .= u[4:6] - cons.con[1].info[:JU]
		c[4:6] .= cons.con[1].info[:JL] - u[4:6]
		c[7:9] .= u[1:3] - cons.con[t].info[:uU]
		c[10:12] .= cons.con[t].info[:uL] - u[1:3]
	elseif t == 10
		c[1:3] .= kinematics(model, x) - cons.con[t].info[:x1]
		c[4:6] .= u[1:3] - cons.con[t].info[:uU]
		c[7:9] .= cons.con[t].info[:uL] - u[1:3]
	elseif t == 20
		c[1:3] .= kinematics(model, x) - cons.con[t].info[:x2]
		c[4:6] .= u[1:3] - cons.con[t].info[:uU]
		c[7:9] .= cons.con[t].info[:uL] - u[1:3]
 	elseif t == 30
		c[1:3] .= kinematics(model, x) - cons.con[t].info[:x3]
		c[4:6] .= u[1:3] - cons.con[t].info[:uU]
		c[7:9] .= cons.con[t].info[:uL] - u[1:3]
	elseif t == T
		c[1:n] .= view(x, 1:6) - cons.con[T].info[:xT]
	else
		c[1:3] .= u[1:3] - cons.con[t].info[:uU]
		c[4:6] .= cons.con[t].info[:uL] - u[1:3]
	end
end

prob = problem_data(model, obj, con_set, copy(x̄), copy(ū), w, h, T,
	n = n, m = m)

# Solve
@time constrained_ddp_solve!(prob,
    max_iter = 1000, max_al_iter = 10,
	ρ_init = 1.0, ρ_scale = 10.0,
	con_tol = 1.0e-3,
	cache = false)

x, u = current_trajectory(prob)
x̄, ū = nominal_trajectory(prob)

u[1][4:6]

plot(hcat([ut[1:3] for ut in ū]...)', linetype = :steppost)

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
