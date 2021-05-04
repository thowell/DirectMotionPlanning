using MeshCat, MeshCatMechanisms#, RigidBodyDynamics
using FileIO, MeshIO, GeometryBasics, CoordinateTransformations, Rotations

urdf_original = joinpath(pwd(),"models/kuka/kuka.urdf")
urdf_new = joinpath(pwd(),"models/kuka/temp/kuka.urdf")

function write_kuka_urdf()
    kuka_mesh_dir = joinpath(pwd(),"models/kuka/meshes")
    temp_dir = joinpath(pwd(),"models/kuka/temp")
    if !isdir(temp_dir)
        mkdir(temp_dir)
    end
    open(urdf_original,"r") do f
        open(urdf_new, "w") do fnew
            for ln in eachline(f)
                pre = findfirst("<mesh filename=",ln)
                post = findlast("/>",ln)
                if !(pre isa Nothing) && !(post isa Nothing)
                    inds = pre[end]+2:post[1]-2
                    pathstr = ln[inds]
                    file = splitdir(pathstr)[2]
                    ln = ln[1:pre[end]+1] * joinpath(kuka_mesh_dir,file) * ln[post[1]-1:end]
                end
                println(fnew,ln)
            end
        end
    end
end

write_kuka_urdf()
