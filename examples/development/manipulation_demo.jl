include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
open(vis)
default_background!(vis)

# box
dir = "/home/taylor/Research/ContactControl.jl/src/dynamics/quadruped/payload_mesh/Box.obj"
obj = MeshFileObject(dir)
setobject!(vis[:box], obj)
tr = Translation(0.495, 0.0, 0.3)
rt = LinearMap(0.2 * RotZ(-0.5 * π) * RotX(1.0 * π))
tform = compose(tr, rt)
settransform!(vis[:box], tform)

include_model("simple_manipulator")

# arm 1
l1 = Cylinder(Point3f0(0, 0, 0),
	Point3f0(0, 0, 1.0),
	convert(Float32, 0.025))
setobject!(vis["l1_1"], l1, MeshPhongMaterial(color = RGBA(0, 0, 0, 1.0)))
l2 = Cylinder(Point3f0(0, 0, 0), Point3f0(0, 0, 1.0),
	convert(Float32, 0.025))
setobject!(vis["l2_1"], l2, MeshPhongMaterial(color = RGBA(0, 0, 0, 1.0)))

setobject!(vis["elbow_1"], Sphere(Point3f0(0),
	convert(Float32, 0.05)),
	MeshPhongMaterial(color = RGBA(0, 0, 0, 1.0)))
setobject!(vis["ee_1"], Sphere(Point3f0(0),
	convert(Float32, 0.05)),
	MeshPhongMaterial(color = RGBA(1.0, 165.0 / 255.0, 0.0, 1.0)))

p1 = [-0.99; -0.01; 0.0]
q1 = [0.41 * π; -0.635 * π]
p_mid = [kinematics_mid(model, q1)[1], 0.0, kinematics_mid(model, q1)[2]] + p1
p_ee = [kinematics_ee(model, q1)[1], 0.0, kinematics_ee(model, q1)[2]] + p1

settransform!(vis["l1_1"], cable_transform(p1, p_mid))
settransform!(vis["l2_1"], cable_transform(p_mid, p_ee))

settransform!(vis["elbow_1"], Translation(p_mid))
settransform!(vis["ee_1"], Translation(p_ee))

# arm 2
setobject!(vis["l1_2"], l1, MeshPhongMaterial(color = RGBA(0, 0, 0, 1.0)))
setobject!(vis["l2_2"], l2, MeshPhongMaterial(color = RGBA(0, 0, 0, 1.0)))

setobject!(vis["elbow_2"], Sphere(Point3f0(0),
	convert(Float32, 0.05)),
	MeshPhongMaterial(color = RGBA(0, 0, 0, 1.0)))
setobject!(vis["ee_2"], Sphere(Point3f0(0),
	convert(Float32, 0.05)),
	MeshPhongMaterial(color = RGBA(1.0, 165.0 / 255.0, 0.0, 1.0)))

p2 = [0.99; -0.01; 0.0]
q2 = [0.4 *  π; 0.825 * π]
p_mid = [kinematics_mid(model, q2)[1], 0.0, kinematics_mid(model, q2)[2]] + p2
p_ee = [kinematics_ee(model, q2)[1], 0.0, kinematics_ee(model, q2)[2]] + p2

settransform!(vis["l1_2"], cable_transform(p2, p_mid))
settransform!(vis["l2_2"], cable_transform(p_mid, p_ee))

settransform!(vis["elbow_2"], Translation(p_mid))
settransform!(vis["ee_2"], Translation(p_ee))

settransform!(vis["/Cameras/default"],
	   compose(Translation(0.0, -95.0, -1.0), LinearMap(RotY(0.0 * π) * RotZ(-π / 2.0))))
setprop!(vis["/Cameras/default/rotated/<object>"], "zoom", 60)

setobject!(vis["box1"], GeometryBasics.HyperRectangle(Vec(0.0, 0.0, 0.0),
		Vec(0.25, 0.5, 0.25)), MeshPhongMaterial(color = RGBA(1.0, 0.0, 0.0, 1.0)))
	settransform!(vis["box1"], Translation(0.51, -0.25, 0))

setobject!(vis["box2"], GeometryBasics.HyperRectangle(Vec(0.0, 0.0, 0.0),
		Vec(0.25, 0.5, 0.25)), MeshPhongMaterial(color = RGBA(1.0, 0.0, 0.0, 1.0)))
	settransform!(vis["box2"], Translation(-0.16, -0.25, 0))
