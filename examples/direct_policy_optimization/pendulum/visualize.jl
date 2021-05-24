using Plots

# TO solution
x̄, ū = unpack(z̄, prob)
@show tf_nom = sum([ū[t][end] for t = 1:T-1]) # total time
t_nominal = range(0, stop = tf_nom, length = T)
plot(t_nominal, hcat(x̄...)', width = 10.0, color = :lawngreen,
	grid = false, label = false, axes = false)
plot(t_nominal[1:end-1], hcat(ū...)[1:1, :]',
	linetype = :steppost, width = 2.0, color = goldenrod_color)

# DPO solution
x, u = unpack(z, prob_dpo.prob.prob.nom)
@show tf_dpo = sum([u[t][end] for t = 1:T-1])
t_dpo = range(0, stop = tf_dpo, length = T)

x_sample = []
u_sample = []
t_sample = []
plt = plot()
for i = 1:2 * model.n
	z_sample = z[prob_dpo.prob.idx.sample[i]]
	_x_sample, _u_sample = unpack(z_sample, prob_dpo.prob.prob.nom)
	tf_sample = sum([_u_sample[t][end] for t = 1:T-1])
	@show tf_sample
	push!(x_sample, _x_sample)
	push!(u_sample, _u_sample)
	push!(t_sample, range(0, stop = tf_sample, length = T))
	# plt = plot!(t_sample[end], hcat(x_sample[end]...)',
	# 	color = :black, width = 5.0, label = "", grid = false)
	plt = plot!(t_sample[end], hcat(u_sample[end]..., u_sample[end][end])[1:1,:]',
		color = :black, width = 5.0, label = "", grid = false, linetype = :steppost)
end
# plt = plot!(t_dpo, hcat(x...)', width = 10.0, color = :lawngreen, label = "", grid = false)

plot!(t_dpo, hcat(u..., u[end])[1:1, :]',
	linetype = :steppost, width = 10.0, color = :lawngreen, label = "", grid = false)

display(plt)
