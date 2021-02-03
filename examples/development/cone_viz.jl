using Colors
using CoordinateTransformations
using GeometryBasics
using MeshCat
using Rotations

# second-order cone
function κ_so(z)
    z1 = z[1:end-1]
    z2 = z[end]

    z_proj = zero(z)

    if norm(z1) <= z2
        z_proj = copy(z)
    elseif norm(z1) <= -z2
        z_proj = zero(z)
    else
        a = 0.5 * (1.0 + z2 / norm(z1))
        z_proj[1:end-1] = a * z1
        z_proj[end] = a * norm(z1)
    end

    return z_proj
end

# visualizer
function second_order_cone_viz(; h = 1.0, trans = 0.1)
    # visualizer
    vis = Visualizer()

    # set empty background
    setvisible!(vis["/Background"], true)
    setprop!(vis["/Background"], "top_color", RGBA(1.0, 1.0, 1.0, 1.0))
    setprop!(vis["/Background"], "bottom_color", RGBA(1.0, 1.0, 1.0, 1.0))
    setvisible!(vis["/Axes"], false)

    # generate cone
    l = h * 1.0
    w = h * 1.45
    pyramid = Pyramid(Point3(0.0, 0.0, 0.0), l, w)

    n = 500
    for i = 1:n
        setobject!(vis["pyramid$i"], pyramid,
            MeshPhongMaterial(color = RGBA(1.0, 0.0, 0.0, trans)))
        settransform!(vis["pyramid$i"],
            compose(Translation(0.0, 0.0, l),
                LinearMap(RotX(π) * RotZ(π * i / n))))
    end

    # generate point
    setobject!(vis["point"], Sphere(Point3f0(0),
        convert(Float32, 0.05)),
        MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

    settransform!(vis["point"], Translation(0.5, 0.5, 0.5))

    return vis
end

function set_point!(vis, p)
    settransform!(vis["point"], Translation(p...))
end

function project_point!(vis, p)
    p_proj = κ_so(p)
    settransform!(vis["point"], Translation(p_proj...))
    return p_proj
end


vis = second_order_cone_viz()
render(vis)

p = [1.0, 0.0, 0.5]

set_point!(vis, p)
p_proj = project_point!(vis, p)
