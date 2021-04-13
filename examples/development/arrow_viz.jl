vis = Visualizer()
render(vis)
orange_mat, blue_mat, black_mat = get_line_material(10)
nc = 1
point_x = [[Point(0.0,0.0,0.0), Point(1.0, 0.0, 0.0)] for i=1:nc]
point_z = [[Point(0.0,0.0,0.0), Point(0.0, 0.0, 1.0)] for i=1:nc]
for i = 1:nc
	setobject!(vis[:test][:impact]["$i"],   MeshCat.Line(point_z[i]))
	setobject!(vis[:test][:friction]["$i"], MeshCat.Line(point_x[i]))
end
settransform!(vis[:test], compose(Translation([1.0; 0.0; 0.0]...), LinearMap(RotY(-1.0 * π / 7.0))))

vis[:test][:impact]["1"]
function visualize!(vis, model, q;
	Δt = 0.1, r = 0.1)

	default_background!(vis)
    setobject!(vis["particle"],
		Rect(Vec(0, 0, 0),Vec(2r, 2r, 2r)),
		MeshPhongMaterial(color = RGBA(1.0, 165.0 / 255.0, 0, 1.0)))

    anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

    for t = 1:length(q)
        MeshCat.atframe(anim, t) do
            settransform!(vis["particle"], Translation(q[t][1:3]...))
        end
    end

    MeshCat.setanimation!(vis, anim)
end

arr_vis = ArrowVisualizer(vis["env/arrow"])
mat = MeshPhongMaterial(color=RGBA(1.0, 0.0, 0.0, 1.0))
setobject!(arr_vis, mat)
settransform!(arr_vis,
	Point(0.0, 0.0, 0.0),
	Vec(0.2, 0.0, 0.0),
	shaft_radius=0.008,
	max_head_radius=0.020)

settransform!(vis["env/arrow"], compose(Translation([1.0; 0.0; 0.0]...), LinearMap(3.0 * RotY(-1.0 * π / 7.0))))

function visualize!(vis, T)

    anim = MeshCat.Animation(convert(Int, floor(1.0 / 0.1)))

    for t = 1:T
        MeshCat.atframe(anim, t) do
			settransform!(vis["env/arrow"], compose(Translation([0.1 * t; 0.0; 0.0]...), LinearMap(0.5 * t * RotY(-1.0 * π / 7.0))))
        end
    end

    MeshCat.setanimation!(vis, anim)
end

visualize!(vis, 10)
