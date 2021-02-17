using Plots
include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)

x̄, ū = unpack(z̄_nom, prob_nominal)
x, u = unpack(z, prob_nominal)

t_nominal = range(0, stop = sum([ū[t][end] for t = 1:T-1]), length = T)
plot(t_nominal[1:end-1], hcat(ū...)[1:end-1, :]', linetype = :steppost)
@show sum([ū[t][end] for t = 1:T-1]) # works when 2.72

# COM traj
xx_nom = [x̄[t][1] for t = 1:T]
zz_nom = [x̄[t][2] for t = 1:T]
x̄_dpo, ū_dpo = unpack(z[prob_dpo.prob.idx.nom], prob_dpo.prob.prob.nom)
t_dpo = range(0, stop = sum([ū_dpo[t][end] for t = 1:T-1]), length = T)

xx = [x̄_dpo[t][1] for t = 1:T]
zz = [x̄_dpo[t][2] for t = 1:T]

plot(xx_nom, zz_nom, color = :cyan)
plot!(xx, zz, color = :orange)

plot(t_nominal, hcat(x̄...)', color = :cyan)
plot!(t_dpo, hcat(x̄_dpo...)', color = :orange)


# orientation tracking
plt_x = plot(t_nominal, hcat(x̄...)[3,:],legend=:topright,color=:red,
    label="",width=2.0,xlabel="time (s)",
    title="Rocket",ylabel="state")
plt_x = plot!(t_sim_nom,hcat(z_lqr...)[3,:],color=:black,label="",
    width=1.0)

plt_x = plot(t_dpo,hcat(x̄_dpo...)[3,:],legend=:topright,color=:red,
    label="",width=2.0,xlabel="time (s)",
    title="Rocket",ylabel="state")
plt_x = plot!(t_sim_dpo,hcat(z_dpo...)[3,:],color=:black,label="",
    width=1.0)

visualize!(vis, model_sl, z_lqr, Δt = dt_sim_nom)
visualize!(vis, model_sl, z_dpo, Δt = dt_sim_dpo)

using PGFPlots
const PGF = PGFPlots

p_traj_nom = PGF.Plots.Linear(hcat(x̄...)[1,:],hcat(x̄...)[2,:],
	mark="none",style="color=cyan, line width = 2pt",legendentry="TO")
p_traj_sample = PGF.Plots.Linear(hcat(x...)[1,:],hcat(x...)[2,:],
	mark="none",style="color=orange, line width = 2pt",legendentry="DPO")

a = Axis([p_traj_nom;p_traj_sample],
    axisEqualImage=false,
    hideAxis=false,
	ylabel="z",
	xlabel="y",
	legendStyle="{at={(0.01,0.99)},anchor=north west}")

# Save to tikz format
PGF.save(joinpath(pwd(),
	"examples/direct_policy_optimization/figures/rocket_traj.tikz"), a, include_preamble=false)

# orientation tracking
p_nom_orientation = PGF.Plots.Linear(t_nom,hcat(x̄...)[3,:],
	mark="none",style="color=cyan, line width = 2pt",legendentry="TO")
p_nom_sim_orientation = PGF.Plots.Linear(t_sim_nom,hcat(z_lqr...)[3,:],
	mark="none",style="color=black, line width = 1pt")

p_sample_orientation = PGF.Plots.Linear(t_dpo,hcat(x...)[3,:],
	mark="none",style="color=orange, line width = 2pt",legendentry="DPO")
p_sample_sim_orientation = PGF.Plots.Linear(t_sim_dpo,hcat(z_dpo...)[3,:],
	mark="none",style="color=black, line width = 1pt")

a = Axis([p_nom_orientation;p_nom_sim_orientation],
    axisEqualImage=false,
    hideAxis=false,
	ylabel="orientation",
	xlabel="time",
	legendStyle="{at={(0.5,0.99)},anchor=north}")

# Save to tikz format
dir = joinpath(pwd(),
	"examples/direct_policy_optimization/figures/rocket_orientation_lqr.tikz")
PGF.save(dir, a, include_preamble=false)

a = Axis([p_sample_orientation;p_sample_sim_orientation],
    axisEqualImage=false,
    hideAxis=false,
	ylabel="orientation",
	xlabel="time",
	legendStyle="{at={(0.5,0.99)},anchor=north}")

# Save to tikz format
dir = joinpath(pwd(),
	"examples/direct_policy_optimization/figures/rocket_orientation_dpo.tikz")
PGF.save(dir, a, include_preamble=false)


# # eigen value analysis
# C = Diagonal(ones(model_sl.nx))[1:model_nom.nx,:]
# # nominal
# A_nom, B_nom = nominal_jacobians(model_nom,X_nom,U_nom,H_nom)
# A_nom_cl = [(A_nom[t] - B_nom[t]*K[t]) for t = 1:T-1]
# sv_nom = [norm.(eigen(A_nom_cl[t]).values) for t = 1:T-1]
# plt_nom = plot(hcat(sv_nom...)',xlabel="time step t",ylabel="eigen value norm",
# 	title="TVLQR nominal model",linetype=:steppost,
# 	ylims=(-3,3),labels="")
#
# # slosh nominal
# X_nom_slosh = [[copy(X_nom[t]);0.0;0.0] for t = 1:T]
# A_nom_slosh, B_nom_slosh = nominal_jacobians(model_sl,X_nom_slosh,U_nom,H_nom)
# A_nom_slosh_cl = [(A_nom_slosh[t] - B_nom_slosh[t]*K[t]*C) for t = 1:T-1]
# sv_nom_slosh = [norm.(eigen(A_nom_slosh_cl[t]).values) for t = 1:T-1]
# plt_nom_slosh = plot(hcat(sv_nom_slosh...)',xlabel="time step t",ylabel="eigen value norm",
# 	title="TVLQR slosh model",linetype=:steppost,
# 	ylims=(-3,3),labels="")
#
# # slosh
# A_dpo, B_dpo = nominal_jacobians(model_nom,X_nom_sample,U_nom_sample,H_nom_sample)
# A_dpo_cl = [(A_dpo[t] - B_dpo[t]*Θ_mat[t]) for t = 1:T-1]
# sv_dpo = [norm.(eigen(A_dpo_cl[t]).values) for t = 1:T-1]
# plt_dpo_nom = plot(hcat(sv_dpo...)',xlabel="time step t",ylabel="singular value",
# 	title="DPO nominal model",linetype=:steppost,
# 	ylims=(-1,3),labels="")
#
# X_dpo_slosh = [[copy(X_nom_sample[t]);0.0;0.0] for t = 1:T]
# A_dpo_slosh, B_dpo_slosh = nominal_jacobians(model_sl,X_dpo_slosh,U_nom_sample,H_nom_sample)
# A_dpo_slosh_cl = [(A_dpo_slosh[t] - B_dpo_slosh[t]*Θ_mat[t]*C) for t = 1:T-1]
# sv_dpo_slosh = [norm.(eigen(A_dpo_slosh_cl[t]).values) for t = 1:T-1]
# plt_dpo_slosh = plot(hcat(sv_dpo_slosh...)',xlabel="time step t",ylabel="eigen value norm",
# 	title="DPO slosh model",linetype=:steppost,
# 	ylims=(-1.0,3.0),labels="")
#
# plot(plt_nom,plt_dpo_nom,layout=(2,1))
#
# plot(plt_nom_slosh,plt_dpo_slosh,layout=(2,1))
#
#
# plot(t_nom,hcat(X_nom...)[1:6,:]',color=:cyan,label=["y" "z" "ϕ"])
# plot!(t_nom_sample, hcat(X_nom_sample...)[1:6,:]',color=:orange,label=["y" "z" "ϕ"])


# Animation plot
vis = Visualizer()
# render(vis)
open(vis)

pts_nom = collect(eachcol(hcat([[-p[1]; -1.0; p[2]] for p in x]...)))
material_nom = LineBasicMaterial(color = colorant"orange", linewidth = 10.0)
setobject!(vis["dpo_traj"], Object(PointCloud(pts_nom), material_nom, "Line"))
setvisible!(vis["dpo_traj"],true)

pts_nom = collect(eachcol(hcat([[-p[1]; 0.01; p[2]] for p in z_dpo]...)))
material_nom = LineBasicMaterial(color = colorant"grey", linewidth = 2.5)
setobject!(vis["dpo_sim"], Object(PointCloud(pts_nom), material_nom, "Line"))
setvisible!(vis["dpo_sim"],true)
# visualize_rocket!(vis, model_slosh, pad_trajectory(z_dpo, zeros(model_slosh.n), 100) , x1, xT,
# 	Δt = dt_sim_dpo, T_off = length(z_dpo) + 100)

pts_nom = collect(eachcol(hcat([[-p[1]; -1.1; p[2]] for p in x̄]...)))
material_nom = LineBasicMaterial(color = colorant"cyan", linewidth = 10.0)
setobject!(vis["to_traj"], Object(PointCloud(pts_nom), material_nom, "Line"))
setvisible!(vis["to_traj"],true)

pts_nom = collect(eachcol(hcat([[-p[1]; 0.01; p[2]] for p in z_lqr]...)))
material_nom = LineBasicMaterial(color = colorant"grey", linewidth = 2.5)
setobject!(vis["to_sim"], Object(PointCloud(pts_nom), material_nom, "Line"))
setvisible!(vis["to_sim"],true)
# visualize_rocket!(vis, model_slosh, pad_trajectory(z_lqr, zeros(model_slosh.n), 100) , x1, xT,
# 	Δt = dt_sim_nom, T_off = length(z_lqr) + 100)

x_ghost = [x[1], x[7], x[15], x[25], x[T]]
visualize_rocket_ghost!(vis, model_slosh, x_ghost)
