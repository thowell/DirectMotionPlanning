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
