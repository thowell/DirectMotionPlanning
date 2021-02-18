using Plots, Distributions, LinearAlgebra, ForwardDiff

n = 100
m = 50

Q = Diagonal(ones(n))

A = rand(m, n)
b = A * randn(n)
z_sol = 1.0 * [2 * Q A'; A zeros(m, m)] \ [zeros(n); b]
# @show norm(lagrangian_gradient(z_sol))
# @show lagrangian(z_sol)

function obj(x)
	x' * Q * x
end

function gradient(x)
	2.0 * Q * x
end

function lagrangian(z)
	x = view(z, 1:n)
	y = view(z, n .+ (1:m))

	obj(x) + y' * (A * x - b)
end

function lagrangian_gradient(z)
	x = view(z, 1:n)
	y = view(z, n .+ (1:m))

	[gradient(x) + A' * y; A * x - b]
end



dir = Diagonal([ones(n); -1.0 * ones(m)])

# adam
function adam(obj, gradient, x;
		max_iter = 1000)

	# history
	J_hist = [obj(x)]

	# parameters
	α = 1.0e-4
	β1 = 0.9
	β2 = 0.999
	ϵ = 10.0^(-8.0)
	m = zero(x)
	v = zero(x)
	m̂ = zero(x)
	v̂ = zero(x)

	# iterate
	for i = 1:max_iter
		g = gradient(x)
		# m .= β1 .* m + (1.0 - β1) .* g
		# v .= β2 .* v + (1.0 - β2) .* (g.^2.0)
		# m̂ .= m ./ (1.0 - β1^i)
		# v̂ .= v ./ (1.0 - β2^i)
		# x .-= α * m̂ ./ (sqrt.(v̂) .+ ϵ)
		x .-= α * dir * g
		push!(J_hist, obj(x))
		println("norm $(norm(gradient(x)))")
		i % 1000 == 0 && println("iter: $i")
	end

	return x, J_hist
end

x_sol, J_hist = adam(lagrangian, lagrangian_gradient, ones(n + m),
	max_iter = 100000)

plt = plot(J_hist,
	xlabel = "iteration", ylabel = "obj.", title = "sgd",
	# label = "α = $α (adam)",
	color = :red, width = 2.0)

display(plt)
