using Plots
include(joinpath(@__DIR__, "quadrotor_broken_propeller.jl"))
include(joinpath(pwd(), "models/visualize.jl"))

x̄, ū = unpack(z̄, prob)
x, u = unpack(z, prob)


@show sum([ū[t][end] for t = 1:T-1])
t_nominal = [0.0, [sum([ū[i][end] for i = 1:t]) for t = 1:T-1]...]
@show sum([u[t][end] for t = 1:T-1])
t_dpo = [0.0, [sum([u[i][end] for i = 1:t]) for t = 1:T-1]...]

plot(t_nominal, hcat(x̄...)[1:3,:]', linetype=:steppost, width = 2.0)
plot(t_nominal[1:end-1], hcat(ū...)', linetype=:steppost, width = 2.0)

plot(t_dpo, hcat(x...)[1:3,:]', linetype=:steppost, width = 2.0)
plot(t_dpo[1:end-1], hcat(u...)', linetype=:steppost, width = 2.0)

vis = Visualizer()
open(vis)
visualize!(vis, model, x̄, Δt = ū[1][end])

# # Plots results
using Plots
# # Position trajectory
x_nom_pos = [X_nom[t][1] for t = 1:T]
y_nom_pos = [X_nom[t][2] for t = 1:T]
# pts = Plots.partialcircle(0,2π,100,r)
# cx,cy = Plots.unzip(pts)
# cx1 = [_cx + xc1 for _cx in cx]
# cy1 = [_cy + yc1 for _cy in cy]
# cx2 = [_cx + xc2 for _cx in cx]
# cy2 = [_cy + yc2 for _cy in cy]
# cx3 = [_cx + xc3 for _cx in cx]
# cy3 = [_cy + yc3 for _cy in cy]
# cx4 = [_cx + xc4 for _cx in cx]
# cy4 = [_cy + yc4 for _cy in cy]
# # cx5 = [_cx + xc5 for _cx in cx]
# # cy5 = [_cy + yc5 for _cy in cy]
#
# plt = plot(Shape(cx1,cy1),color=:red,label="",linecolor=:red)
# # plt = plot!(Shape(cx2,cy2),color=:red,label="",linecolor=:red)
# # plt = plot!(Shape(cx3,cy3),color=:red,label="",linecolor=:red)
# # plt = plot!(Shape(cx4,cy4),color=:red,label="",linecolor=:red)
# # # plt = plot(Shape(cx5,cy5),color=:red,label="",linecolor=:red)
#
# for i = 1:N
#     x_sample_pos = [X_sample[i][t][1] for t = 1:T]
#     y_sample_pos = [X_sample[i][t][2] for t = 1:T]
#     plt = plot!(x_sample_pos,y_sample_pos,aspect_ratio=:equal,
#         color=:cyan,label= i != 1 ? "" : "sample")
# end
# display(plt)
plt = plot()
plt = scatter!(x_nom_pos,y_nom_pos,aspect_ratio=:equal,xlabel="x",ylabel="y",width=4.0,label="TO",color=:purple,legend=:topleft)
x_sample_pos = [X_nom_sample[t][1] for t = 1:T]
y_sample_pos = [X_nom_sample[t][2] for t = 1:T]
plt = plot!(x_sample_pos,y_sample_pos,aspect_ratio=:equal,width=4.0,label="DPO",color=:orange,legend=:bottomright)
#
# savefig(plt,joinpath(@__DIR__,"results/quadrotor_trajectory.png"))
#
# Control
plt = plot(t_nominal[1:T-1],Array(hcat(U_nom...))',color=:purple,width=2.0,
    title="quad",xlabel="time (s)",ylabel="control",
    legend=:bottom,linetype=:steppost)
plt = plot!(t_sample[1:T-1],Array(hcat(U_nom_sample...))',color=:orange,
    width=2.0,linetype=:steppost)
savefig(plt,joinpath(@__DIR__,"results/quadrotor_control.png"))

# # Samples
X_sample
# State samples
plt1 = plot(title="Sample states",legend=:bottom,xlabel="time (s)");
for i = 1:N
    # t_sample = zeros(T)
    # for t = 2:T
    #     t_sample[t] = t_sample[t-1] + H_nom_sample[t-1]
    # end
    plt1 = plot!(hcat(X_sample[i]...)',label="");
end
plt1 = plot!(hcat(X_nom_sample...)',color=:red,width=2.0,
    label=["nominal" "" ""])
display(plt1)
# savefig(plt1,joinpath(@__DIR__,"results/quadrotor_sample_states.png"))
#
# # Control samples
# plt2 = plot(title="Sample controls",xlabel="time (s)",legend=:bottom);
# for i = 1:N
#     t_sample = zeros(T)
#     for t = 2:T
#         t_sample[t] = t_sample[t-1] + H_nom_sample[t-1]
#     end
#     plt2 = plot!(t_sample[1:end-1],hcat(U_sample[i]...)',label="",
#         linetype=:steppost);
# end
# plt2 = plot!(t_sample[1:end-1],hcat(U_nom_sample...)',color=:red,width=2.0,
#     label=["nominal" ""],linetype=:steppost)
# display(plt2)
# savefig(plt2,joinpath(@__DIR__,"results/quadrotor_sample_controls.png"))




# plot states
plt_x = plot(t_nom,hcat(X_nom...)[1:model.nx,:]',
	legend=:topright,color=:red,
    label="",width=2.0,xlabel="time (s)",
    title="Quadrotor",ylabel="state")
plt_x = plot!(t_sim_nom,hcat(z_tvlqr1...)[1:model.nx,:]',color=:black,label="",
    width=1.0)
plt_x = plot!(t_sim_nom,hcat(z_tvlqr2...)[1:model.nx,:]',color=:black,label="",
    width=1.0)
plt_x = plot!(t_sim_nom,hcat(z_tvlqr3...)[1:model.nx,:]',color=:black,label="",
    width=1.0)
plt_x = plot!(t_sim_nom,hcat(z_tvlqr4...)[1:model.nx,:]',color=:black,label="",
    width=1.0)

plt_x = plot(t_nom_sample,hcat(X_nom_sample...)[1:model.nx,:]',
	legend=:topright,color=:red,
    label="",width=2.0,xlabel="time (s)",
    title="Quadrotor",ylabel="state")
plt_x = plot!(t_sim_nom_sample,hcat(z_sample1...)[1:model.nx,:]',color=:black,label="",
    width=1.0)
plt_x = plot!(t_sim_nom_sample,hcat(z_sample2...)[1:model.nx,:]',color=:black,label="",
	width=1.0)
plt_x = plot!(t_sim_nom_sample,hcat(z_sample3...)[1:model.nx,:]',color=:black,label="",
    width=1.0)
plt_x = plot!(t_sim_nom_sample,hcat(z_sample4...)[1:model.nx,:]',color=:black,label="",
    width=1.0)

# plot COM
plot_traj = plot(hcat(X_nom...)[1,:],hcat(X_nom...)[2,:],
	legend=:topright,color=:red,
    label="",width=2.0,xlabel="y",ylabel="z",
    title="Quadrotor")
plot_traj = plot!(hcat(z_tvlqr1...)[1,:],hcat(z_tvlqr1...)[2,:],
	color=:black,
    label="",width=1.0)
plot_traj = plot!(hcat(z_tvlqr2...)[1,:],hcat(z_tvlqr2...)[2,:],
	color=:black,
	label="",width=1.0)
plot_traj = plot!(hcat(z_tvlqr3...)[1,:],hcat(z_tvlqr3...)[2,:],
	color=:black,
    label="",width=1.0)
plot_traj = plot!(hcat(z_tvlqr4...)[1,:],hcat(z_tvlqr4...)[2,:],
	color=:black,
    label="",width=1.0)

plot_traj = plot(hcat(X_nom_sample...)[1,:],hcat(X_nom_sample...)[2,:],
	legend=:topright,color=:red,
    label="",width=2.0,xlabel="y",ylabel="z",
    title="Quadrotor")
plot_traj = plot!(hcat(z_sample1...)[1,:],hcat(z_sample1...)[2,:],
	color=:black,
    label="",width=1.0)
plot_traj = plot!(hcat(z_sample2...)[1,:],hcat(z_sample2...)[2,:],
	color=:black,
    label="",width=1.0)
plot_traj = plot!(hcat(z_sample3...)[1,:],hcat(z_sample3...)[2,:],
	color=:black,
    label="",width=1.0)
plot_traj = plot!(hcat(z_sample4...)[1,:],hcat(z_sample4...)[2,:],
	color=:black,
    label="",width=1.0)

plot(t_nom[1:end-1],hcat(U_nom...)',
	linetype=:steppost,width=2.0)
plot!(t_sim_nom[1:end-1],hcat(u_tvlqr1...)',
	linetype=:steppost,color=:black,width=1.0)

plot(t_nom[1:end-1],hcat(U_nom_sample...)',
	linetype=:steppost,width=2.0)
plot!(t_sim_nom[1:end-1],hcat(u_sample1...)',
	linetype=:steppost,color=:black,width=1.0)


# state tracking
(Jx_tvlqr1 + Jx_tvlqr2 + Jx_tvlqr3 + Jx_tvlqr4)/4.0
(Jx_sample1 + Jx_sample2 + Jx_sample3 + Jx_sample4)/4.0

# control tracking
(Ju_tvlqr1 + Ju_tvlqr2 + Ju_tvlqr3 + Ju_tvlqr4)/4.0
(Ju_sample1 + Ju_sample2 + Ju_sample3 + Ju_sample4)/4.0

# objective value
(J_tvlqr1 + J_tvlqr2 + J_tvlqr3 + J_tvlqr4)/4.0
(J_sample1 + J_sample2 + J_sample3 + J_sample4)/4.0

using PGFPlots
const PGF = PGFPlots

# TO trajectory
p_u1_nom = PGF.Plots.Linear(t_nominal[1:end],hcat(ū..., ū[end])[1,:],
    mark="none",style="const plot, color=cyan, line width=2pt, solid")
p_u2_nom = PGF.Plots.Linear(t_nominal[1:end],hcat(ū..., ū[end])[2,:],
    mark="none",style="const plot, color=cyan, line width=2pt, solid")
p_u3_nom = PGF.Plots.Linear(t_nominal[1:end],hcat(ū..., ū[end])[3,:],
    mark="none",style="const plot, color=cyan, line width=2pt, solid")
p_u4_nom = PGF.Plots.Linear(t_nominal[1:end],hcat(ū..., ū[end])[4,:],
    mark="none",style="const plot, color=cyan, line width=2pt, solid")

p_u1_dpo = PGF.Plots.Linear(t_dpo[1:end],hcat(u..., u[end])[1,:],
    mark="none",style="const plot, color=orange, line width=2pt, solid")
p_u2_dpo = PGF.Plots.Linear(t_dpo[1:end],hcat(u..., u[end])[2,:],
    mark="none",style="const plot, color=orange, line width=2pt, solid")
p_u3_dpo = PGF.Plots.Linear(t_dpo[1:end],hcat(u..., u[end])[3,:],
    mark="none",style="const plot, color=orange, line width=2pt, solid")
p_u4_dpo = PGF.Plots.Linear(t_dpo[1:end],hcat(u..., u[end])[4,:],
    mark="none",style="const plot, color=orange, line width=2pt, solid")


a1 = Axis([p_u1_nom;p_u1_dpo
    ],
    xmin=0, ymin=0,
    hideAxis=false,
	ylabel="u1",
	xlabel="time",
	)
a2 = Axis([p_u2_nom;p_u2_dpo
    ],
    xmin=0, ymin=0,
    hideAxis=false,
	ylabel="u2",
	xlabel="time",
	)
a3 = Axis([p_u3_nom;p_u3_dpo
    ],
    xmin=0, ymin=0,
    hideAxis=false,
	ylabel="u3",
	xlabel="time",
	)
a4 = Axis([p_u4_nom;p_u4_dpo
    ],
    xmin=0, ymin=0,
    hideAxis=false,
	ylabel="u4",
	xlabel="time",
	)

# Save to tikz format
dir = joinpath(pwd(), "examples/direct_policy_optimization/figures")
PGF.save(joinpath(dir,"quad_prop_u1.tikz"), a1, include_preamble=false)
PGF.save(joinpath(dir,"quad_prop_u2.tikz"), a2, include_preamble=false)
PGF.save(joinpath(dir,"quad_prop_u3.tikz"), a3, include_preamble=false)
PGF.save(joinpath(dir,"quad_prop_u4.tikz"), a4, include_preamble=false)


# # visualize
include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
visualize!(vis,model,x̄,Δt=ū[1][end])
visualize!(vis,model,x,Δt=u[1][end])

pts_nom = collect(eachcol(hcat([z[1:3] for z in x̄]...)))
material_nom = LineBasicMaterial(color = colorant"cyan", linewidth = 10.0)
setobject!(vis["com_traj_nom"], Object(PointCloud(pts_nom), material_nom, "Line"))
setvisible!(vis["com_traj_nom"],false)

pts_nom = collect(eachcol(hcat([z[1:3] for z in z_lqr1]...)))
material_nom = LineBasicMaterial(color = colorant"gray", linewidth = 5.0)
setobject!(vis["com_traj_lqr1"], Object(PointCloud(pts_nom), material_nom, "Line"))
setvisible!(vis["com_traj_lqr1"],false)

pts_nom = collect(eachcol(hcat([z[1:3] for z in z_lqr2]...)))
material_nom = LineBasicMaterial(color = colorant"gray", linewidth = 5.0)
setobject!(vis["com_traj_lqr2"], Object(PointCloud(pts_nom), material_nom, "Line"))
setvisible!(vis["com_traj_lqr2"],false)

pts_nom = collect(eachcol(hcat([z[1:3] for z in z_lqr3]...)))
material_nom = LineBasicMaterial(color = colorant"gray", linewidth = 5.0)
setobject!(vis["com_traj_lqr3"], Object(PointCloud(pts_nom), material_nom, "Line"))
setvisible!(vis["com_traj_lqr3"],false)

pts_nom = collect(eachcol(hcat([z[1:3] for z in z_lqr4]...)))
material_nom = LineBasicMaterial(color = colorant"gray", linewidth = 5.0)
setobject!(vis["com_traj_lqr4"], Object(PointCloud(pts_nom), material_nom, "Line"))
setvisible!(vis["com_traj_lqr4"],false)

pts_nom = collect(eachcol(hcat([z[1:3] for z in x]...)))
material_nom = LineBasicMaterial(color = colorant"orange", linewidth = 10.0)
setobject!(vis["com_traj_dpo"], Object(PointCloud(pts_nom), material_nom, "Line"))
setvisible!(vis["com_traj_dpo"],true)

pts_nom = collect(eachcol(hcat([z[1:3] .+ 1.0e-3 for z in z_dpo1]...)))
material_nom = LineBasicMaterial(color = colorant"gray", linewidth = 5.0)
setobject!(vis["com_traj_dpo1"], Object(PointCloud(pts_nom), material_nom, "Line"))
setvisible!(vis["com_traj_dpo1"],true)

pts_nom = collect(eachcol(hcat([z[1:3] .+ 1.0e-3 for z in z_dpo2]...)))
material_nom = LineBasicMaterial(color = colorant"gray", linewidth = 5.0)
setobject!(vis["com_traj_dpo2"], Object(PointCloud(pts_nom), material_nom, "Line"))
setvisible!(vis["com_traj_dpo2"],true)

pts_nom = collect(eachcol(hcat([z[1:3] .+ 1.0e-3 for z in z_dpo3]...)))
material_nom = LineBasicMaterial(color = colorant"gray", linewidth = 5.0)
setobject!(vis["com_traj_dpo3"], Object(PointCloud(pts_nom), material_nom, "Line"))
setvisible!(vis["com_traj_dpo3"],true)

pts_nom = collect(eachcol(hcat([z[1:3] .+ 1.0e-3 for z in z_dpo4]...)))
material_nom = LineBasicMaterial(color = colorant"gray", linewidth = 5.0)
setobject!(vis["com_traj_dpo4"], Object(PointCloud(pts_nom), material_nom, "Line"))
setvisible!(vis["com_traj_dpo4"],true)

# for (k,z) in enumerate([z_tvlqr1,z_tvlqr2,z_tvlqr3,z_tvlqr4])
# 	i = k
# 	q_to = z
# 	for t = 1:3:T_sim
# 		setobject!(vis["traj_to$t$i"], Sphere(Point3f0(0),
# 			convert(Float32,0.025)),
# 			MeshPhongMaterial(color=RGBA(128.0/255.0,128.0/255.0,128.0/255.0,1.0)))
# 		settransform!(vis["traj_to$t$i"], Translation((q_to[t][1],q_to[t][2],q_to[t][3])))
# 		setvisible!(vis["traj_to$t$i"],false)
# 	end
# end
# q_to_nom = X_nom
# for t = 1:T
# 	setobject!(vis["traj_to_nom$t"], Sphere(Point3f0(0),
# 		convert(Float32,0.05)),
# 		MeshPhongMaterial(color=RGBA(0.0,255.0/255.0,255.0/255.0,1.0)))
# 	settransform!(vis["traj_to_nom$t"], Translation((q_to_nom[t][1],q_to_nom[t][2],q_to_nom[t][3])))
# 	setvisible!(vis["traj_to_nom$t"],false)
# end
#
# for (k,z) in enumerate([z_sample1,z_sample2,z_sample3,z_sample4])
# 	i = k
# 	q_dpo = z
# 	for t = 1:3:T_sim
# 		setobject!(vis["traj_dpo$t$i"], Sphere(Point3f0(0),
# 			convert(Float32,0.025)),
# 			MeshPhongMaterial(color=RGBA(128.0/255.0,128.0/255.0,128.0/255.0,1.0)))
# 		settransform!(vis["traj_dpo$t$i"], Translation((q_dpo[t][1],q_dpo[t][2],q_dpo[t][3])))
# 		setvisible!(vis["traj_dpo$t$i"],true)
# 	end
# end
#
# q_dpo_nom = X_nom_sample
# for t = 1:T
# 	setobject!(vis["traj_dpo_nom$t"], Sphere(Point3f0(0),
# 		convert(Float32,0.05)),
# 		MeshPhongMaterial(color=RGBA(255.0/255.0,127.0/255.0,0.0,1.0)))
# 	settransform!(vis["traj_dpo_nom$t"], Translation((q_dpo_nom[t][1],q_dpo_nom[t][2],q_dpo_nom[t][3])))
# 	setvisible!(vis["traj_dpo_nom$t"],true)
# end
#
obj_path = joinpath(pwd(),
   "models/quadrotor/drone.obj")
 mtl_path = joinpath(pwd(),
   "models/quadrotor/drone.mtl")
ctm = ModifiedMeshFileObject(obj_path,mtl_path,scale=1.0)
setobject!(vis["drone2"],ctm)
settransform!(vis["drone2"], compose(Translation(x[10][1:3]),LinearMap(MRP(x[10][4:6]...)*RotX(pi/2.0))))

# open(vis)
