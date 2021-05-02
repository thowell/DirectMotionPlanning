"""
      Satellite

      Orientation represented with
      Modified Rodrigues Parameters.
"""

struct Satellite{I, T} <: Model{I, T}
      n::Int
      m::Int
      d::Int

      J      # inertia matrix
end

function f(model::Satellite, z, u, w)
      # states
      r = view(z, 1:3)
      ω = view(z, 4:6)

      # controls
      τ = view(u, 1:3)

      SVector{6}([0.25 * ((1.0 - r' * r) * ω - 2.0 * cross(ω, r) + 2.0 * (ω' * r) * r);
                  model.J \ (τ - cross(ω, model.J * ω))])
end

function kinematics(model::Satellite, q)
	p = @SVector [1.0, 0.0, 0.0]
	k = MRP(view(q, 1:3)) * p
	return k
end

# model
n, m, d = 6, 3, 0
J = Diagonal(@SVector[1.0, 2.0, 3.0])

model = Satellite{Midpoint, FixedTime}(n, m, d, J)

# visuals
function visualize!(vis, p::Satellite, q; Δt = 0.1)
	setvisible!(vis["/Background"], true)
	setprop!(vis["/Background"], "top_color", RGBA(0.0, 0.0, 0.0, 1.0))
	setprop!(vis["/Background"], "bottom_color", RGBA(0.0, 0.0, 0.0, 1.0))
	setvisible!(vis["/Axes"], false)
	setvisible!(vis["/Grid"], false)

    setobject!(vis[:satellite],
    	Rect(Vec(-0.25, -0.25, -0.25),Vec(0.5, 0.5, 0.5)),
    	MeshPhongMaterial(color = RGBA(1.0, 1.0, 1.0, 1.0)))

    arrow_x = ArrowVisualizer(vis[:satellite][:arrow_x])
    mat = MeshPhongMaterial(color=RGBA(1.0, 0.0, 0.0, 1.0))
    setobject!(arrow_x, mat)
    settransform!(arrow_x,
    	Point(0.0, 0.0, 0.0),
    	Vec(0.75, 0.0, 0.0),
    	shaft_radius=0.05,
    	max_head_radius=0.1)

    arrow_y = ArrowVisualizer(vis[:satellite][:arrow_y])
    mat = MeshPhongMaterial(color=RGBA(0.0, 1.0, 0.0, 1.0))
    setobject!(arrow_y, mat)
    settransform!(arrow_y,
    	Point(0.0, 0.0, 0.0),
    	Vec(0.0, 0.75, 0.0),
    	shaft_radius=0.05,
    	max_head_radius=0.1)

    arrow_z = ArrowVisualizer(vis[:satellite][:arrow_z])
    mat = MeshPhongMaterial(color=RGBA(0.0, 0.0, 1.0, 1.0))
    setobject!(arrow_z, mat)
    settransform!(arrow_z,
    	Point(0.0, 0.0, 0.0),
    	Vec(0.0, 0.0, 0.75),
    	shaft_radius=0.05,
    	max_head_radius=0.1)

    Δt = 0.1
    anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

     for t = 1:length(x̄)
    	 MeshCat.atframe(anim, t) do
    		 settransform!(vis["satellite"],
    			   compose(Translation(0.0 * [-0.25; -0.25; -0.25]...),
    					 LinearMap(MRP(x̄[t][1:3]...))))
    	 end
     end


     MeshCat.setanimation!(vis, anim)
end
