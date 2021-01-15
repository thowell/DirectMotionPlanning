n = 2
μ0 = zeros(n)
P0 = Diagonal(ones(n))

x0 = resample(μ0, P0, 1.0)

β = 10.0
μ1 = sample_mean(x0)
P1 = sample_covariance(x0, β)

A = Diagonal(ones(n))
B = zeros(n)

x1 = []
u0 = rand(1)
for j = 1:2n
	push!(x1, A * x0[j] + B * u0[1])
end

A' * P0 * A
β = 10.0
μ1 = sample_mean(x1)
P1 = sample_covariance(x1, β)
