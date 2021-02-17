function cable_transform(y, z)
    v1 = [0.0, 0.0, 1.0]
    # if norm(z) > norm(y)
    #     v2 = y[1:3,1] - z[1:3,1]
    # else
    #     v2 = z[1:3,1] - y[1:3,1]
    # end
    v2 = y[1:3,1] - z[1:3,1]
    normalize!(v2)
    ax = cross(v1, v2)
    ang = acos(v1'*v2)
    R = AngleAxis(ang, ax...)

    if any(isnan.(R))
        R = I
    else
        nothing
    end

    compose(Translation(z), LinearMap(R))
end

function ModifiedMeshFileObject(obj_path::String, material_path::String;
        scale::T = 0.1) where {T}
    obj = MeshFileObject(obj_path)
    rescaled_contents = rescale_contents(obj_path, scale = scale)
    material = select_material(material_path)
    mod_obj = MeshFileObject(
        rescaled_contents,
        obj.format,
        material,
        obj.resources,
        )
    return mod_obj
end

function rescale_contents(obj_path::String; scale::T = 0.1) where T
    lines = readlines(obj_path)
    rescaled_lines = copy(lines)
    for (k,line) in enumerate(lines)
        if length(line) >= 2
            if line[1] == 'v'
                stringvec = split(line, " ")
                vals = map(x -> parse(Float64, x), stringvec[2:end])
                rescaled_vals = vals .* scale
                rescaled_lines[k] = join([stringvec[1]; string.(rescaled_vals)], " ")
            end
        end
    end
    rescaled_contents = join(rescaled_lines, "\r\n")
    return rescaled_contents
end

function select_material(material_path::String)
    mtl_file = open(material_path)
    mtl = read(mtl_file, String)
    return mtl
end

function default_background!(vis)
    setvisible!(vis["/Background"], true)
    setprop!(vis["/Background"], "top_color", RGBA(1.0, 1.0, 1.0, 1.0))
    setprop!(vis["/Background"], "bottom_color", RGBA(1.0, 1.0, 1.0, 1.0))
    setvisible!(vis["/Axes"], false)
end

function pad_trajectory(x, shift, T_shift)
	[[x[1] + shift for i = 1:T_shift]..., [_x + shift for _x in x]..., [x[end] + shift for i = 1:T_shift]...]
end
