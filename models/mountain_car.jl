"""
    Mountain Car

    https://en.wikipedia.org/wiki/Mountain_car_problem
"""

struct MountainCar{I, T} <: Model{I, T}
    n::Int
    m::Int
    d::Int

    gravity
    mass
    steepness
    friction

    xl
    xu
    ul
    uu
end

function fd(model::MountainCar{Discrete, FixedTime}, x⁺, x, u, w, h, t)
    x⁺ - [x[1] + x[2] * h;
          x[2] + (model.gravity * model.mass * cos(model.steepness * x[1]) 
            + u[1] / model.mass - model.friction * x[2]) * h]
end

function fd(model::MountainCar{Discrete, FixedTime}, x, u, w, h, t)
    [x[1] + x[2] * h;
     x[2] + (model.gravity * model.mass * cos(model.steepness * x[1]) 
        + u[1] / model.mass - model.friction * x[2]) * h]
end

n, m, d = 2, 1, 0
gravity = 9.81
mass = 0.2
steepness = 3.0
friction = 0.3
xl = [-1.2; -1.5]
xu = [0.5; 1.5]
ul = [-2.0]
uu = [2.0]
model = MountainCar{Discrete, FixedTime}(n, m, d,
    gravity, mass, steepness, friction, xl, xu, ul, uu)

function visualize!(vis, model::Car, x;
       Δt = 0.1, traj = false)

	default_background!(vis)

	obj_path = joinpath(pwd(),"models/cybertruck/cybertruck.obj")
	mtl_path = joinpath(pwd(),"models/cybertruck/cybertruck.mtl")

	m = ModifiedMeshFileObject(obj_path,mtl_path,scale=0.05)
	setobject!(vis["car"],m)

   	anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

    for t = 1:length(x)
        MeshCat.atframe(anim,t) do
			settransform!(vis["car"],
				compose(Translation(x[t][1], x[t][2], 0.0),
					LinearMap(RotZ(x[t][3] + pi) * RotX(pi / 2.0))))
        end
    end

	if traj
		pts_nom = collect(eachcol(hcat(x...)))
		material_nom = LineBasicMaterial(color = colorant"orange", linewidth = 3.0)
		setobject!(vis["traj"], Object(PointCloud(pts_nom), material_nom, "Line"))
	end

	settransform!(vis["/Cameras/default"],
		compose(Translation(0.0, 0.0, 0.0),LinearMap(RotY(-pi/2.5))))

    MeshCat.setanimation!(vis, anim)
end
