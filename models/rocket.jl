"""
	Rocket
"""

abstract type Rocket{I, T} <: Model{I, T} end

struct RocketNominal{I, T} <: Rocket{I, T}
	n::Int
	m::Int
	d::Int

	g  # gravity
	m1 # mass
	l1 # length
	J  # inertia
end

struct RocketSlosh{I, T} <: Rocket{I, T}
	n::Int
	m::Int
	d::Int

	g  # gravity
	m1 # mass
	l1 # length
	J  # inertia

	m2 # mass of fuel
	l2 # length of fuel pendulum
	l3 # length from rocket COM to fuel pendulum
end

g = 9.81 # gravity
m1 = 1.0 # mass of rocket
l1 = 0.5 # length from COM to thruster
J = 1.0 / 12.0 * m1 * (2.0 * l1)^2.0 # inertia of rocket

m2 = 0.1 # mass of pendulum
l2 = 0.1 # length from COM to pendulum
l3 = 0.1 # length of pendulum

# models
m = 2

# nominal model
n_nominal = 6
model_nominal = RocketNominal{Midpoint, FixedTime}(n_nominal, m, n_nominal, g, m1, l1, J)

# slosh model
n_slosh = 8
model_slosh = RocketSlosh{Midpoint, FixedTime}(n_slosh, m, n_slosh, g, m1 - m2, l1, J, m2, l2, l3)

function k_thruster(model::Rocket, x)
	y, z, θ = x[1:3]
	py = y + model.l1 * sin(θ)
	pz = z - model.l1 * cos(θ)

	return [py;
		    pz]
end

function jacobian_thruster(model::RocketNominal, x)
	y, z, θ = x[1:3]

	return [1.0 0.0 model.l1 * cos(θ);
			0.0 1.0 model.l1 * sin(θ)]
end

function jacobian_thruster(model::RocketSlosh, x)
	y, z, θ = x[1:3]

	return [1.0 0.0 model.l1 * cos(θ) 0.0;
			0.0 1.0 model.l1 * sin(θ) 0.0]
end

function k_mass(model::RocketSlosh, x)
	yp = x[1] + l2 * sin(x[3]) + l3 * sin(x[4])
	zp = x[2] - l2 * cos(x[3]) - l3 * cos(x[4])

	return [yp;
			zp]
end

function jacobian_mass(model::RocketSlosh, x)
	return [1.0 0.0 l2 * cos(x[3]) l3 * cos(x[4]);
	        0.0 1.0 l2 * sin(x[3]) l3 * sin(x[4])]
end

function lagrangian(model::RocketNominal, q, q̇)
	return (0.5 * model.m1 * (q̇[1]^2 + q̇[2]^2.0) + 0.5 * model.J * q̇[3]^2.0
			- model.m1 * model.g * q[2])
end

function lagrangian(model::RocketSlosh, q, q̇)
	zp = k_mass(model, q)[2]
	vp = jacobian_mass(model, q) * q̇

	return (0.5 * model.m1 * (q̇[1]^2 + q̇[2]^2) + 0.5 * model.J * q̇[3]^2.0
			- model.m1 * model.g * q[2]
			+ 0.5 * model.m2 * vp' * vp
			- model.m2 * model.g * zp)
end

function dLdq(model::Rocket, q, q̇)
	Lq(x) = lagrangian(model, x, q̇)
	ForwardDiff.gradient(Lq, q)
end

function dLdq̇(model::Rocket, q, q̇)
	Lq̇(x) = lagrangian(model, q, x)
	ForwardDiff.gradient(Lq̇, q̇)
end

function f(model::Rocket, x, u, w)
	nq = convert(Int, floor(model.n / 2.0))
	q = x[1:nq]
	q̇ = x[nq .+ (1:nq)]
	tmp_q(z) = dLdq̇(model, z, q̇)
	tmp_q̇(z) = dLdq̇(model, q, z)
	[q̇;
	 ForwardDiff.jacobian(tmp_q̇,q̇) \ (-1.0 * ForwardDiff.jacobian(tmp_q, q) * q̇
	 	+ dLdq(model, q, q̇)
		+ jacobian_thruster(model, q)' * u[1:2])]
end

function state_output(model::RocketSlosh, x)
	x[collect([1, 2, 3, 5, 6, 7])]
end

function visualize_rocket!(vis, model::Rocket, x, x1, xT;
       Δt = 0.1, version = :slosh_nom, T_off = length(x))

	T = length(x)

	default_background!(vis)

	body = Cylinder(Point3f0(0.0, 0.0, -1.0 * model.l1),
		Point3f0(0.0, 0.0, 3.0 * model.l1),
		convert(Float32, 0.225))

	setobject!(vis["rocket"]["body"], body,
		MeshPhongMaterial(color = RGBA(1.0, 0.0, 0.0, 1.0)))

	obj_rocket = joinpath(pwd(), "models/rocket/space_x_booster.obj")
	mtl_rocket = joinpath(pwd(), "models/rocket/space_x_booster.mtl")
	ctm = ModifiedMeshFileObject(obj_rocket,mtl_rocket,scale=1.0)
	setobject!(vis["rocket"]["falcon9"], ctm)

	settransform!(vis["rocket"]["falcon9"],
		compose(Translation(([-xT[1] + 4.0; -4.0; xT[2] - 1.0])...),
		LinearMap(RotY(1.0 * xT[3]) * RotZ(pi) * RotX(pi / 2.0))))

	settransform!(vis["rocket"],
		compose(Translation(-1.0 * x1[1], 0.0, x1[2]),
		LinearMap(RotY(x1[3]))))


	setvisible!(vis["rocket"]["falcon9"], true)
	obj_platform = joinpath(pwd(), "models/rocket/space_x_platform.obj")
	mtl_platform = joinpath(pwd(), "models/rocket/space_x_platform.mtl")

	obj_platform = joinpath(pwd(), "models/rocket/space_x_platform.obj")
	mtl_platform = joinpath(pwd(), "models/rocket/space_x_platform.mtl")
	ctm_platform = ModifiedMeshFileObject(obj_platform,mtl_platform,scale=1.0)
	setobject!(vis["platform"],ctm_platform)
	settransform!(vis["platform"], compose(Translation(0.0,2.5,-0.85),LinearMap(RotZ(pi)*RotX(pi/2))))

	anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

    for t = 1:T
        MeshCat.atframe(anim,t) do
			if t >= T_off
				setvisible!(vis["rocket"]["body"], false)
			else
				setvisible!(vis["rocket"]["body"], true)
			end
			settransform!(vis["rocket"],
				compose(Translation(-1.0 * x[t][1], 0.0, x[t][2]),
				LinearMap(RotY(x[t][3]))))
        end
    end

	settransform!(vis["/Cameras/default"], compose(Translation(0.0, 20.0, -1.0),
		LinearMap(RotZ(pi / 2.0))))
	setvisible!(vis["/Grid"], false)

    MeshCat.setanimation!(vis, anim)
end

function visualize_rocket_ghost!(vis, model::Rocket, x)

	default_background!(vis)

	body = Cylinder(Point3f0(0.0, 0.0, -1.0 * model.l1),
		Point3f0(0.0, 0.0, 3.0 * model.l1),
		convert(Float32, 0.225))
	rkt_offset = [3.9,-6.35,0.2]
	y_shift = 2.35
	obj_rocket = joinpath(pwd(), "models/rocket/space_x_booster.obj")
	mtl_rocket = joinpath(pwd(), "models/rocket/space_x_booster.mtl")
	ctm = ModifiedMeshFileObject(obj_rocket,mtl_rocket,scale=1.0)

	for (t, _x) in enumerate(x)
		println("t= $t")
		setobject!(vis["rocket_$t"]["body"], body,
			MeshPhongMaterial(color = RGBA(1.0, 0.0, 0.0, 1.0)))
		setvisible!(vis["rocket_$t"]["body"], t == length(x) ? false : true)


		setobject!(vis["rocket_$t"]["falcon9"], ctm)
		settransform!(vis["rocket_$t"]["falcon9"],
			compose(Translation(([-x[end][1]- 0.1225; y_shift; x[end][2]-1.15] + rkt_offset)...),
			LinearMap(RotY(1.0 * x[end][3]) * RotZ(pi) * RotX(pi / 2.0))))
		setvisible!(vis["rocket_$t"]["falcon9"], true)

		settransform!(vis["rocket_$t"],
			compose(Translation(-1.0 * _x[1], 0.0, _x[2]),
			LinearMap(RotY(_x[3]))))
	end

	obj_platform = joinpath(pwd(), "models/rocket/space_x_platform.obj")
	mtl_platform = joinpath(pwd(), "models/rocket/space_x_platform.mtl")

	obj_platform = joinpath(pwd(), "models/rocket/space_x_platform.obj")
	mtl_platform = joinpath(pwd(), "models/rocket/space_x_platform.mtl")
	ctm_platform = ModifiedMeshFileObject(obj_platform,mtl_platform,scale=1.0)

	setobject!(vis["platform"],ctm_platform)
	settransform!(vis["platform"], compose(Translation(0.0,2.5,-0.85),LinearMap(RotZ(pi)*RotX(pi/2))))

	setvisible!(vis["/Grid"], false)

	settransform!(vis["/Cameras/default"], compose(Translation(0.0, 20.0, -1.0),
		LinearMap(RotZ(pi / 2.0))))
end

function visualize_simple!(vis, model::Rocket, x;
       Δt = 0.1)

	default_background!(vis)

	l1 = Cylinder(Point3f0(0.0, 0.0, -1.0 * model.l1),
		Point3f0(0.0, 0.0, 3.0 * model.l1),
		convert(Float32, 0.25))

    setobject!(vis["rocket"], l1,
		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

   	anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

    for t = 1:length(x)
        MeshCat.atframe(anim,t) do
			settransform!(vis["rocket"],
				compose(Translation(-1.0 * x[t][1], 0.0, x[t][2]),
				LinearMap(RotY(x[t][3]))))
        end
    end

    settransform!(vis["/Cameras/default"], compose(Translation(0.0, 25.0, -1.0),
		LinearMap(RotZ(pi / 2.0))))

    MeshCat.setanimation!(vis, anim)
end
