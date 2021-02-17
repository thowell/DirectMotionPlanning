using Plots

x̄, ū = unpack(z̄, prob)

# Position trajectory
Px = [x̄[t][1] for t = 1:T]
Py = [x̄[t][2] for t = 1:T]

pts = Plots.partialcircle(0.0, 2.0 * π, 100, 0.1)
cx, cy = Plots.unzip(pts)
cx1 = [_cx + circles[1][1] for _cx in cx]
cy1 = [_cy + circles[1][2] for _cy in cy]
cx2 = [_cx + circles[2][1] for _cx in cx]
cy2 = [_cy + circles[2][2] for _cy in cy]
cx3 = [_cx + circles[3][1] for _cx in cx]
cy3 = [_cy + circles[3][2] for _cy in cy]
cx4 = [_cx + circles[4][1] for _cx in cx]
cy4 = [_cy + circles[4][2] for _cy in cy]

plt = plot(Shape(cx1, cy1), color = :black, label = "", linecolor = :black)
plt = plot!(Shape(cx2, cy2), color = :black, label = "", linecolor = :black)
plt = plot!(Shape(cx3, cy3), color = :black, label = "", linecolor = :black)
plt = plot!(Shape(cx4, cy4), color = :black, label = "", linecolor = :black)
plt = plot!(Px, Py, aspect_ratio = :equal, xlabel = "x", ylabel = "y",
    width = 4.0, label = "TO", color = :cyan, legend = :topleft)

# DPO
x, u = unpack(z, prob_dpo.prob.prob.nom)


# Position trajectory
Px_dpo = [x[t][1] for t = 1:T]
Py_dpo = [x[t][2] for t = 1:T]
plt = plot!(Px_dpo, Py_dpo, width = 4.0, label = "DPO", color = :orange)

# Position trajectory (samples)
x_sample = []
for i = 1:2 * model.n
	z_sample = z[prob_dpo.prob.idx.sample[i]]
	_x_sample, u_sample = unpack(z_sample, prob_dpo.prob.prob.nom)
	push!(x_sample, _x_sample)
	Px_sample = [_x_sample[t][1] for t = 1:T]
	Py_sample = [_x_sample[t][2] for t = 1:T]
	plt = plot!(Px_sample, Py_sample, width = 2.0, label = "sample", color = :magenta)
end
display(plt)

# using PGFPlots
# const PGF = PGFPlots
#
# # TO trajectory
# p_nom = PGF.Plots.Linear(hcat(x̄...)[1,:], hcat(x̄...)[2,:],
#     mark = "none",
# 	style = "color=cyan, line width=2pt, solid",
# 	legendentry = "TO")
#
# # DPO trajectory
# p_dpo = PGF.Plots.Linear(hcat(x...)[1,:], hcat(x...)[2,:],
#     mark = "none",
# 	style = "color=orange, line width=2pt, solid",
# 	legendentry = "DPO")
#
# # obstacles
# pc1 = PGF.Plots.Circle(circles[1]...,
# 	style = "color=black, fill=black")
# pc2 = PGF.Plots.Circle(circles[2]...,
# 	style = "color=black, fill=black")
# pc3 = PGF.Plots.Circle(circles[3]...,
# 	style = "color=black, fill=black")
# pc4 = PGF.Plots.Circle(circles[4]...,
# 	style = "color=black, fill=black")
#
# a = Axis([
#     p_nom;
#     p_dpo;
# 	pc1;
# 	pc2;
# 	pc3;
# 	pc4],
#     xmin = -0.4, ymin = -0.1, xmax = 1.4, ymax = 1.1,
#     axisEqualImage = true,
#     hideAxis = true,
# 	# ylabel = "y",
# 	# xlabel = "x",
# 	legendStyle = "{at={(0.01, 0.99)}, anchor = north west}")
#
# # Save to tikz format
# PGF.save(joinpath(@__DIR__, "car_obstacles.tikz"), a, include_preamble = false)

# Animation
include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
# render(vis)
open(vis)
default_background!(vis)
settransform!(vis["/Cameras/default"],
	compose(Translation(0.0, 0.0, -1.0),LinearMap(RotY(-pi/2.5))))
shift = [-0.5; -0.5; 0.0]

for i = 1:4
    cyl = Cylinder(Point3f0(circles[i][1] - 0.5,circles[i][2] - 0.5,0), Point3f0(circles[i][1] - 0.5,circles[i][2] -0.5,0.1),convert(Float32,0.035))
    setobject!(vis["cyl$i"],cyl,MeshPhongMaterial(color=RGBA(0,0,0,1.0)))
end

q_to = deepcopy(x̄)

# interpolate traj
T = length(q_to)
T_sim = 4 * T
times = [(t - 1) * h for t = 1:T-1]
tf = h * (T-1)
t_sim = range(0, stop = tf, length = T_sim)
t_ctrl = range(0, stop = tf, length = T)
dt_sim = tf / (T_sim - 1)
A_state = hcat(q_to...)
z_cubic = zero(q_to[1])

for t = 1:T_sim
	for i = 1:length(q_to[1])
		interp_cubic = CubicSplineInterpolation(t_ctrl, A_state[i,:])
		z_cubic[i] = interp_cubic(t_sim[t])
	end
	setobject!(vis["traj_to$t"],
		Cylinder(Point3f0(0.0, 0.0, -0.001),
			Point3f0(0.0, 0.0, 0.0),
			convert(Float32,0.065)),
			MeshPhongMaterial(color=RGBA(0.0,255.0/255.0,255.0/255.0,1.0)))

		# HyperSphere(Point3f0(0),
		# convert(Float32,0.075)),
		# MeshPhongMaterial(color=RGBA(0.0,255.0/255.0,255.0/255.0,1.0)))
	settransform!(vis["traj_to$t"], Translation((z_cubic[1] - 0.5,z_cubic[2] -0.5,0.0)))
	setvisible!(vis["traj_to$t"],false)
end

q_dpo = deepcopy(x)
A_state = hcat(q_dpo...)
z_cubic = zero(q_dpo[1])
for t = 1:T_sim
	for i = 1:length(q_dpo[1])
		interp_cubic = CubicSplineInterpolation(t_ctrl, A_state[i,:])
		z_cubic[i] = interp_cubic(t_sim[t])
	end
	setobject!(vis["traj_dpo$t"], Cylinder(Point3f0(0.0, 0.0, -0.001),
		Point3f0(0.0, 0.0, 0.001),
		convert(Float32,0.065)),
		MeshPhongMaterial(color=RGBA(255.0/255.0,127.0/255.0,0.0,1.0)))
	settransform!(vis["traj_dpo$t"], Translation((z_cubic[1] - 0.5,z_cubic[2] - 0.5,0.0)))
	setvisible!(vis["traj_dpo$t"],true)
end

for i = 1:2 * model.n
	# q_dpo = deepcopy(x_sample[i])
	A_state = hcat(q_dpo...)
	z_cubic = zero(q_dpo[1])
	for t = 1:T_sim
		for i = 1:length(q_dpo[1])
			interp_cubic = CubicSplineInterpolation(t_ctrl, A_state[i,:])
			z_cubic[i] = interp_cubic(t_sim[t])
		end
		setobject!(vis["traj_sample$t$i"], Cylinder(Point3f0(0.0, 0.0, -0.001),
			Point3f0(0.0, 0.0, 0.000),
			convert(Float32,0.065)),
			MeshPhongMaterial(color=RGBA(255.0/255.0,0.0,255.0,1.0)))
		settransform!(vis["traj_sample$t$i"], Translation((z_cubic[1] - 0.5,z_cubic[2] - 0.5,0.0)))
		setvisible!(vis["traj_sample$t$i"],false)
	end
end



q = [[x[1] for i = 1:25]..., x..., [x[T] for i = 1:25]...]
q = [[x̄[1] for i = 1:25]..., x̄..., [x̄[T] for i = 1:25]...]

obj_path = joinpath(pwd(),"/home/taylor/Research/motion_planning/models/cybertruck/cybertruck.obj")
mtl_path = joinpath(pwd(),"/home/taylor/Research/motion_planning/models/cybertruck/cybertruck.mtl")
m = ModifiedMeshFileObject(obj_path, mtl_path, scale = 0.05)
setobject!(vis["car"], m)
settransform!(vis["car"], compose(Translation([q[t][1];q[t][2];0.0] + shift),LinearMap(RotZ(q[t][3]+pi)*RotX(pi/2.0))))

anim = MeshCat.Animation(convert(Int, floor(1.0 / h)))

for t = 1:length(q)
	MeshCat.atframe(anim,t) do
		settransform!(vis["car"],
			compose(Translation(q[t][1] - 0.5, q[t][2] - 0.5, 0.0),
				LinearMap(RotZ(q[t][3] + pi) * RotX(pi / 2.0))))
	end
end
MeshCat.setanimation!(vis, anim)


ctm = ModifiedMeshFileObject(obj_path, mtl_path, scale=0.05)
t = 1
setobject!(vis["ct1"],ctm)
settransform!(vis["ct1"], compose(Translation([q[t][1];q[t][2];0.0] + shift),LinearMap(RotZ(q[t][3]+pi)*RotX(pi/2.0))))
setvisible!(vis["ct1"],true)
t = 1
setobject!(vis["ct2"],ctm)
settransform!(vis["ct2"], compose(Translation([q[t][1];q[t][2];0.0] + shift),LinearMap(RotZ(q[t][3]+pi)*RotX(pi/2.0))))
setvisible!(vis["ct2"],true)
t = 20#17#20
setobject!(vis["ct3"],ctm)
settransform!(vis["ct3"], compose(Translation([q[t][1];q[t][2];0.0] + shift),LinearMap(RotZ(q[t][3]+pi)*RotX(pi/2.0))))
setvisible!(vis["ct3"],true)

t = 34#36#34
setobject!(vis["ct4"],ctm)
settransform!(vis["ct4"], compose(Translation([q[t][1];q[t][2];0.0] + shift),LinearMap(RotZ(q[t][3]+pi)*RotX(pi/2.0))))
setvisible!(vis["ct4"],true)
t = T
setobject!(vis["ct5"],ctm)
settransform!(vis["ct5"], compose(Translation([q[t][1];q[t][2];0.0] + shift),LinearMap(RotZ(q[t][3]+pi)*RotX(pi/2.0))))
setvisible!(vis["ct5"],true)

settransform!(vis["/Cameras/default"],
	compose(Translation(0.0, 0.0, 1.0),LinearMap(RotY(-pi/2.5))))

open(vis)
