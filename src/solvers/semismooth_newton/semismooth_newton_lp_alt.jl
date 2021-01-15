using LinearAlgebra, ForwardDiff, Distributions
using Convex, SCS, ECOS
using IterativeSolvers

# problem setup
p = 60
N = 30
x_sol = max.(0.0, randn(p))
G = rand(N, p)
h = G * x_sol
ν_sol = randn(N)
λ_sol = rand(p)
λ_sol[x_sol .> 0.0] .= 0.0
c = -G' * ν_sol + λ_sol

"Convex.jl"
x = Variable(p)
prob = minimize(c' * x)
prob.constraints += h - G * x == 0.0
prob.constraints += x >= 0.0

@time solve!(prob, ECOS.Optimizer)

@show prob.status
@show prob.constraints
@show x.value
# @show prob.constraints[1].dual
# @show prob.constraints[2].dual
A = [-1.0 * Diagonal(ones(p)); G; -G]
b = [zeros(p); h; -h]
m = 2 * N + p
k = p + m + 1

Q = Array([zeros(p, p) A' c;
           -A zeros(m, m) b;
           -c' -b'      0.0])

function κ(z)
    # z_proj = zero(z)
    # z_proj[1:N] = z[1:N]
    # z_proj[N .+ (1:p)] = max.(0.0, z[N .+ (1:p)])

    z_proj = max.(0.0, z)
    return z_proj
    # return z
end

κ(randn(10))

function Pc(z)
    z_proj = zero(z)
    z_proj[1:p] = z[1:p]
    z_proj[p .+ (1:m)] = κ(z[p .+ (1:m)])
    z_proj[p + m + 1] = max(0.0, z[p + m + 1])

    return z_proj
end

function F(z)
    u = z[1:k]
    v = z[k .+ (1:k)]

    [Q * u - v;
     u - Pc(u - v)]
end

function Ju(z)
    u = z[1:k]
    v = z[k .+ (1:k)]

    JP = zeros(m, m)
    dif = u[p .+ (1:m)] - v[p .+ (1:m)]
    for i = 1:m
        if dif[i] >= 0.0
            JP[i, i] = 1.0
        else
            JP[i, i] = 0.0
        end
    end

    if z[k] - z[2k] >= 0.0
        ℓ = 1.0
    else
        ℓ = 0.0
    end

    [zeros(p, p) zeros(p, m) zeros(p) I zeros(p, m) zeros(p);
     zeros(m, p) I-JP zeros(m) zeros(m, p) JP zeros(m);
     [zeros(1, p) zeros(1, m) 1.0-ℓ zeros(1, p) zeros(1, m) ℓ]]
end

function J(z)
    [Q -I;
     Ju(z)]
end

u = zeros(k)
v = zeros(k)

u[end] = 1.0
v[end] = 1.0

z = [u; v]
z = rand(2k)
F(z)
rank(Ju(z))
J(z)
rank(ForwardDiff.jacobian(F, z))
norm(J(z) - ForwardDiff.jacobian(F, z), Inf)
gmres(J(z), -F(z))
# (J(z)' * J(z)) \ (J(z)' * F(z))
# Δ = zero(z)
# Δ[k] = 1.0
# Δ[2k] = 1.0
# Δ[3k] = 1.0
# gmres!(Δ, J(z), -1.0 * F(z), abstol = 1.0)

function solve()
    u = zeros(k)
    v = zeros(k)
    u[end] = 1.0
    v[end] = 1.0

    z = [u; v]
    # z = rand(3k)
    # Δ = zero(z)

    extra_iters = 0

    for i = 1:500
        _F = F(z)
        _J = J(z)
        Δ = gmres(_J, 1.0 * _F, abstol = 1.0e-12, maxiter = i + extra_iters)

        iter = 0
        α = 1.0
        while norm(F(z - α * Δ))^2.0 >= (1.0 - 0.001 * α) * norm(_F)^2.0 && α > 1.0e-4
            α = 0.5 * α
            # println("   α = $α")
            iter += 1
            if iter > 100
                @error "line search fail"
                # return z
                x = z[(1:p)]
                y = z[p .+ (1:m)]
                τ = z[k]
                κ = z[2k]
                println("τ = $τ")
                println("κ = $κ")
                return x ./ τ
            end
        end

        if α <= 1.0e-4
            extra_iters += 1
        end

        println("iter ($i) - norm: $(norm(F(z)))")

        z .-= α * Δ
    end

    # return z
    x = z[(1:p)]
    y = z[p .+ (1:m)]
    τ = z[k]
    κ = z[2k]
    println("τ = $τ")
    println("κ = $κ")
    return x ./ τ
end

x_sol = solve()

x_sol
b - A * x_sol
norm(x_sol - x.value)
