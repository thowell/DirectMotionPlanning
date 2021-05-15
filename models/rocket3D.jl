"""
      Rocket3D
"""

struct Rocket3D{I, T} <: Model{I, T}
      n::Int
      m::Int
      d::Int

      mass          # mass
      inertia       # inertia matrix
      inertia_inv   # inertia matrix inverse
      gravity       # gravity
      length        # length (com to thruster)
end

function f(model::Rocket3D, z, u, w)
      # states
      x = view(z,1:3)
      r = view(z,4:6)
      v = view(z,7:9)
      ω = view(z,10:12)

      # force in body frame
      F = view(u, 1:3)

      # torque in body frame
      τ = @SVector [model.length * u[2],
                    -model.length * u[1],
                    0.0]

      SVector{12}([v;
                   0.25 * ((1.0 - r' * r) * ω - 2.0 * cross(ω, r) + 2.0*(ω' * r) * r);
                   model.gravity + (1.0 / model.mass) * MRP(r[1], r[2], r[3]) * F;
                   model.inertia_inv * (τ - cross(ω, model.inertia * ω))])
end

n, m, d = 12, 3, 0

mass = 1.0
len = 1.0
inertia = Diagonal(@SVector[1.0 / 12.0 * mass * len^2.0, 1.0 / 12.0 * mass * len^2.0, 1.0e-5])
inertia_inv = Diagonal(@SVector[1.0 / (1.0 / 12.0 * mass * len^2.0), 1.0 / (1.0 / 12.0 * mass * len^2.0), 1.0 / (1.0e-5)])

model = Rocket3D{Midpoint, FixedTime}(n, m, d,
                  mass,
                  inertia,
                  inertia_inv,
                  @SVector[0.0, 0.0, -9.81],
                  len)

function visualize!(vis, p::Rocket3D, q; Δt = 0.1, mesh = true)
	default_background!(vis)

	if mesh
		obj_rocket = joinpath(pwd(), "models/starship/Starship.obj")
		mtl_rocket = joinpath(pwd(), "models/starship/Starship.mtl")
		ctm = ModifiedMeshFileObject(obj_rocket, mtl_rocket, scale=1.0)
		setobject!(vis["rocket"]["starship"], ctm)

		settransform!(vis["rocket"]["starship"],
			compose(Translation(0.0, 0.0, -p.length),
				LinearMap(0.25 * RotY(0.0) * RotZ(0.5 * π) * RotX(0.5 * π))))
	else
		body = Cylinder(Point3f0(0.0, 0.0, -1.0 * model.length),
			Point3f0(0.0, 0.0, 1.0 * model.length),
			convert(Float32, 0.15))

		setobject!(vis["rocket"], body,
			MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))
			anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))
	end

	anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

	for t = 1:length(q)
	    MeshCat.atframe(anim, t) do
	        settransform!(vis["rocket"],
	              compose(Translation(q[t][1:3]),
	                    LinearMap(MRP(q[t][4:6]...) * RotX(0.0))))
	    end
	end

	# settransform!(vis["/Cameras/default"], compose(Translation(0.0, 0.0, 0.0),
	# LinearMap(RotZ(pi/2))))
	MeshCat.setanimation!(vis, anim)
end

# include(joinpath(pwd(), "models/visualize.jl"))
#
# vis = Visualizer()
# render(vis)
# q0 = zeros(model.n)
#
# # simulation test
# u = [0.0; 0.0; 1.0]
# T = 10
# h = 0.1
# x_hist = [q0]
#
# for t = 1:T
# 	push!(x_hist, fd(model, x_hist[end], u, zeros(model.d), h, t))
# end
#
# visualize!(vis, model, x_hist, Δt = 0.1)
