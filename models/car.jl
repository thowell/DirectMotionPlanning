"""
    Car

    Unicycle model (http://lavalle.pl/planning/)
"""

struct Car{I, T} <: Model{I, T}
    n::Int
    m::Int
    d::Int
end

function f(model::Car, x, u, w)
    @SVector [u[1] * cos(x[3]),
              u[1] * sin(x[3]),
              u[2]]
end

n, m, d = 3, 2, 3
model = Car{Midpoint, FixedTime}(n, m, d)

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
