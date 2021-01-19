using LinearAlgebra, ForwardDiff, Distributions
using Convex, SCS, ECOS
using IterativeSolvers

function κ_no(z)
    return max.(0.0, z)
end

function Jκ_no(z)
    x = zero(z)
    for i = 1:length(x)
        if z[i] >= 0.0
            x[i] = 1.0
        end
    end
    return Diagonal(x)
end

function κ_soc(z)
    z1 = z[1:end-1]
    z2 = z[end]

    z_proj = zero(z)

    if norm(z1) <= z2
        z_proj = copy(z)
    elseif norm(z1) <= -z2
        z_proj = zero(z)
    else
        a = 0.5 * (1.0 + z2 / norm(z1))
        z_proj[1:end-1] = a * z1
        z_proj[end] = a * norm(z1)
    end
    return z_proj
end

function Jκ_soc(z)
    z1 = z[1:end-1]
    z2 = z[end]
    m = length(z)

    if norm(z1) <= z2
        return Diagonal(ones(m))
    elseif norm(z1) <= -z2
        return Diagonal(zeros(m))
    else
        D = zeros(m, m)
        for i = 1:m
            if i < m
                D[i, i] = 0.5 + 0.5 * z2 / norm(z1) - 0.5 * z2 * ((z1[i])^2.0) / norm(z1)^3.0
            else
                D[i, i] = 0.5
            end
            for j = 1:m
                if j > i
                    if j < m
                        D[i, j] = -0.5 * z2 * z1[i] * z1[j] / norm(z1)^3.0
                        D[j, i] = -0.5 * z2 * z1[i] * z1[j] / norm(z1)^3.0
                    elseif j == m
                        D[i, j] = 0.5 * z1[i] / norm(z1)
                        D[j, i] = 0.5 * z1[i] / norm(z1)
                    end
                end
            end
        end
        return D
    end
end

# problem setup
p = 10
n = p + 1
Σ = Diagonal(rand(p))
Σ_sqrt = sqrt(Σ)
norm(Σ_sqrt * Σ_sqrt - Σ)

c = [zeros(p); 1.0]
G1 = [2.0 * Σ_sqrt zeros(p); zeros(1, p) -1.0]
h = [zeros(p); 1.0]
q = [zeros(p); 1.0]
z = 1.0
G2 = [ones(p)' 0.0]
G3 = [-ones(p)' 0.0]

A = [-G1; -q'; G2; G3]
b = [h; z; 1.0; -1.0]
m = size(A, 1)

"Convex.jl"
x = Variable(n)
prob = minimize(c' * x)
prob.constraints += norm(G1 * x + h) <= q' * x + z
prob.constraints += G2 * x <= 1.0
prob.constraints += G3 * x <= -1.0

@time solve!(prob, ECOS.Optimizer)

@show prob.status
@show prob.constraints[1].dual
@show x.value
# @show prob.constraints[1].dual
# @show prob.constraints[2].dual
# A = [-1.0 * Diagonal(ones(n)); G; -G]
# b = [zeros(n); h; -h]
# m = 2 *_m  + p
k = n + m + 1
n + 2
Q = Array([zeros(n, n) A' c;
           -A zeros(m, m) b;
           -c' -b'      0.0])

function Pc(z)
    z_proj = zero(z)
    z_proj[1:n] = z[1:n]
    z_proj[n .+ (1:p+2)] = κ_soc(z[n .+ (1:p+2)])
    z_proj[n + p + 2 + 1] = κ_no(z[n + p + 2 + 1])
    z_proj[n + p + 2 + 2] = κ_no(z[n + p + 2 + 2])
    z_proj[n + m + 1] = max(0.0, z[n + m + 1])

    return z_proj
end

function F(z)
    ũ = z[1:k]
    u = z[k .+ (1:k)]
    v = z[2 * k .+ (1:k)]

    [(I + Q) * ũ - (u + v);
     u - Pc(ũ - v);
     ũ - u]
end

function Ju(z)
    ũ = z[1:k]
    u = z[k .+ (1:k)]
    v = z[2k .+ (1:k)]

    JP = zeros(m, m)
    JP[1:p+2, 1:p+2] = Jκ_soc(ũ[n .+ (1:m)][1:p+2] - v[n .+ (1:m)][1:p+2])
    JP[p+2 .+ (1:1), p+2 .+ (1:1)] = Jκ_no(ũ[n .+ (1:m)][p+2 .+ (1:1)] - v[n .+ (1:m)][p+2 .+ (1:1)])
    JP[p+2 + 1 .+ (1:1), p+2 + 1 .+ (1:1)] = Jκ_no(ũ[p+2 .+ (1:m)][p+2 + 1 .+ (1:1)] - v[p+2 .+ (1:m)][p+2 + 1 .+ (1:1)])

    if z[k] - z[3k] >= 0.0
        ℓ = 1.0
    else
        ℓ = 0.0
    end

    [-I zeros(n, m) zeros(n) I zeros(n, m) zeros(n) I zeros(n, m) zeros(n);
     zeros(m, n) -JP zeros(m) zeros(m, n) I zeros(m) zeros(m, n) JP zeros(m);
     [zeros(1, n) zeros(1, m) -ℓ zeros(1, n) zeros(1, m) 1.0 zeros(1, n) zeros(1, m) ℓ]]
end

function J(z)
    [(I + Q) -I -I
     Ju(z);
     I -I zeros(k, k)]
end

ũ = zeros(k)
u = zeros(k)
v = zeros(k)
ũ[end] = 1.0
u[end] = 1.0
v[end] = 1.0

z = [ũ; u; v]
z = rand(3k)
F(z)
rank(Ju(z))
J(z)
rank(ForwardDiff.jacobian(F, z))
norm(J(z) - ForwardDiff.jacobian(F, z), Inf)
gmres(J(z), -F(z))

function solve()
    ũ = zeros(k)
    u = zeros(k)
    v = zeros(k)
    ũ[end] = 1.0
    u[end] = 1.0
    v[end] = 1.0

    z = [ũ; u; v]

    extra_iters = 0

    for i = 1:100
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
                x = z[k .+ (1:n)]
                y = z[k + n .+ (1:m)]
                τ = z[k + k]
                κ = z[3k]
                println("τ = $τ")
                println("κ = $κ")
                return x ./ τ, z
            end
        end

        if α <= 1.0e-4
            extra_iters += 1
        end

        println("iter ($i) -> norm: $(norm(F(z)))")

        z .-= α * Δ
    end

    # return z
    x = z[k .+ (1:n)]
    y = z[k + n .+ (1:m)]
    τ = z[k + k]
    κ = z[3k]
    println("τ = $τ")
    println("κ = $κ")
    return x ./ τ, z
end

x_sol, z_sol = solve()
@show norm(x_sol - x.value)

y_sol = z_sol[1:k] - z_sol[2k .+ (1:k)]
