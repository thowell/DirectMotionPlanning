using Random, Distributions, Plots
include_model("double_integrator")

T = 10
h = 1.0
u_dist = Distributions.MvNormal(zeros(model.m),
	Diagonal(1.0e-1 * ones(model.m)))
u_hist = rand(u_dist, T-1)

x1 = [-1.0; 0.0]
x_hist = [x1]

for t = 1:T-1
	w = zeros(model.d)
	x⁺ = propagate_dynamics(model, x_hist[end], u_hist[t], w, h, t;
			solver = :levenberg_marquardt,
			tol_r = 1.0e-8, tol_d = 1.0e-6)
	push!(x_hist, x⁺)
end

plot(hcat(x_hist...)')
plot(hcat(u_hist...)')

# LQR
A, B = get_dynamics(model)
Q = @SMatrix [1.0 0.0; 0.0 0.0]
R = @SMatrix [1.0]

function objective(x, u)
	T = length(x)
	J = 0.0
	for t = 1:T
		J += x[t]' * Q * x[t]
		t < T & continue
		J += u[t]' * R * u[t]
	end
	return J / T
end

K, P = tvlqr([A for t = 1:T-1], [B for t = 1:T-1],
	[Q for t = 1:T], [R for t = 1:T-1])

K_vec = [vec(K[t]) for t = 1:T-1]

plot(hcat(K_vec...)')
K_inf = copy(K[1])

x1 = [-1.0; 0.0]
x_hist = [x1]
u_hist = []

for t = 1:T-1
	w = zeros(model.d)
	# push!(u_hist, -K[t] * x_hist[end])
	push!(u_hist, -K_inf * x_hist[end])

	x⁺ = propagate_dynamics(model, x_hist[end], u_hist[end], w, h, t;
			solver = :levenberg_marquardt,
			tol_r = 1.0e-8, tol_d = 1.0e-6)

	push!(x_hist, x⁺)
end

plot(hcat(x_hist...)')
plot(hcat(u_hist...)')

J = objective(x_hist, u_hist) # 2.297419

# Gaussian policy
sigmoid(z) = 1.0 / (1.0 + exp(-z))
ds(z) = sigmoid(z) * (1.0 - sigmoid(z))
# plot(range(-10.0, stop = 10.0, length = 100), ds.(range(-10.0, stop = 10.0, length = 100)))
get_μ(θ, x) = θ[1:2]' * x
get_σ(θ, x) = ds(θ[3:4]' * x)

function policy(θ, x, a)
	μ = get_μ(θ, x)
	σ = get_σ(θ, x)

	1.0 / (sqrt(2.0 * π) * σ) * exp(-0.5 * ((a - μ) / (σ))^2.0)
end

function rollout(θ, x1, model, T;
		w = zeros(model.d))

	# initialize trajectories
	x_hist = [x1]
	u_hist = []
	∇logπ_hist = []

	π_dist = []

	for t = 1:T-1

		# policy distribution
		μ = get_μ(θ, x_hist[end])
		σ = get_σ(θ, x_hist[end])

		π_dist = Distributions.Normal(μ, σ)

		# sample control
		push!(u_hist, rand(π_dist, 1)[1])

		# gradient of policy
		logp(z) = log.(policy(z, x_hist[end], u_hist[end]))
		push!(∇logπ_hist, ForwardDiff.gradient(logp, θ))

		# step dynamics
		x⁺ = propagate_dynamics(model, x_hist[end], u_hist[t], w, h, t;
				solver = :levenberg_marquardt,
				tol_r = 1.0e-8, tol_d = 1.0e-6)
		push!(x_hist, x⁺)
	end

	# evalute trajectory cost
	J = objective(x_hist, u_hist)

	return x_hist, u_hist, ∇logπ_hist, J
end

θ = 0.1 * rand(m)
# rollout(θ, x1, model, T;
# 		w = zeros(model.d))

function reinforce(m, x1, model, T;
	max_iter = 1,
	batch_size = 1,
	solver = :sgd,
	J_avg_min = Inf)

	Random.seed!(1)

	# parameters
	θ = 1.0e-5 * rand(m)

	# cost history
	J_hist = [1000.0]
	J_avg = [1000.0]

	if solver == :adam
		# Adam
		α = 0.0001
		β1 = 0.9
		β2 = 0.999
		ϵ = 10.0^(-8.0)
		mom = zeros(m)
		vel = zeros(m)

	else
		# sgd
		α = 1.0e-6
	end

	for i = 1:max_iter
		∇logπ_hist = []
		J = []

		for j = 1:batch_size
			_x_hist, _u_hist, _∇logπ_hist, _J = rollout(θ, copy(x1), model, T)
			push!(∇logπ_hist, _∇logπ_hist)
			push!(J, _J)
		end

		G = (1 / batch_size) .* sum([sum(g) for g in ∇logπ_hist]) .* (mean(J) - J_avg[end])#0.0 * x1' * P[1] * x1)

		if solver == :adam
			mom = β1 .* mom + (1.0 - β1) .* G
			vel = β2 .* vel + (1.0 - β2) .* (G.^2.0)
			m̂ = mom ./ (1.0 - β1^i)
			v̂ = vel ./ (1.0 - β2^i)
			θ .-= α * m̂ ./ (sqrt.(v̂) .+ ϵ)
		else
			θ .-= α * G
		end

		push!(J_hist, mean(J))
		push!(J_avg, J_avg[end] + (mean(J) - J_avg[end]) / length(J_hist))

		if J_avg[end] < J_avg_min
			return θ, J_hist, J_avg
		end

		if i % 1000 == 0
			println("i: $i")
			println("	cost: $(mean(J))")
			println("	cost (run. avg.): $(J_avg[end])")
		end
	end

	return θ, J_hist, J_avg
end

θ, J, J_avg = reinforce(4, x1, model, T,
	max_iter = 1e6,
	batch_size = 10,
	solver = :adam,
	J_avg_min = 0.3)

plot(log.(J[1:100:end]), xlabel = "epoch (100 ep.)", label = "cost")
plot!(log.(J_avg[1:100:end]), xlabel = "epoch (100 ep.)", label = "cost (run. avg.)",
	width = 2.0)

x_hist,  = rollout(θ, x1, model, T)

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
visualize!(vis, model, x_hist, Δt = h)
