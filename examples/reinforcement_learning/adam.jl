using Plots, Distributions, LinearAlgebra

n = 10

function obj(x)
	x' * x
end

function gradient(x, w = zero(x))
	2.0 * x + w
end

# sgd
max_iter = 1000
w_dist = Distributions.MvNormal(zeros(n),
	Diagonal(10.0 * ones(n)))
w_i = rand(w_dist, max_iter)
plt = plot();

for α in [0.1, 0.01, 0.001, 0.0001]
	x = 1.0 * ones(n)
	J_hist = [obj(x)]

	for i = 1:max_iter
		x .-= α * gradient(x, vec(w_i[:, i]))
		push!(J_hist, obj(x))
	end

	plt = plot!(J_hist,
		xlabel = "iteration", ylabel = "obj.",
		label = "α = $α (sgd)")
end

display(plt)

# adam
x = ones(n)


function adam(obj, gradient, x;
		max_iter = 1000)

	# problem size
	n = length(x)

	# history
	J_hist = [obj(x)]

	# parameters
	α = 0.01
	β1 = 0.9
	β2 = 0.999
	ϵ = 10.0^(-8.0)
	m = zero(x)
	v = zero(x)
	m̂ = zero(x)
	v̂ = zero(x)

	# iterate
	for i = 1:max_iter
		g = gradient(x, vec(w_i[:, i]))
		m .= β1 .* m + (1.0 - β1) .* g
		v .= β2 .* v + (1.0 - β2) .* (g.^2.0)
		m̂ .= m ./ (1.0 - β1^i)
		v̂ .= v ./ (1.0 - β2^i)
		x .-= α * m̂ ./ (sqrt.(v̂) .+ ϵ)
		push!(J_hist, obj(x))
	end

	return x, J_hist
end

x_sol, J_hist = adam(obj, gradient, ones(n))

plt = plot!(J_hist,
	xlabel = "iteration", ylabel = "obj.", title = "sgd",
	label = "α = $α (adam)",
	color = :red, width = 2.0)

display(plt)
