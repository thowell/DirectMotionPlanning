using LinearAlgebra, ForwardDiff, Distributions
using Convex, SCS, ECOS
using IterativeSolvers

# problem setup
"Convex.jl"
P = [1.0 0.0 0.0;
     0.0 1.0 0.0]
D = [P; -P]
v = [1.0; 1.0; 0.0]
p = 4
c = D * v
y = -1.0e-8
β = Variable(p)
prob = minimize(c' * β)
prob.constraints += β' * ones(p) <= y
prob.constraints += β >= 0.0

@time solve!(prob, ECOS.Optimizer)

@show prob.status
@show prob.constraints
@show D' * β.value
# @show prob.constraints[1].dual
# @show prob.constraints[2].dual
A = [-1.0 * Diagonal(ones(p)); ones(1,p)]
b = [zeros(p); y]
m = size(A, 1)
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

function Pc(z)
    z_proj = zero(z)
    z_proj[1:p] = z[1:p]
    z_proj[p .+ (1:m)] = κ(z[p .+ (1:m)])
    z_proj[p + m + 1] = max(0.0, z[p + m + 1])

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
    dif = ũ[p .+ (1:m)] - v[p .+ (1:m)]
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

    [-I zeros(p, m) zeros(p) I zeros(p, m) zeros(p) I zeros(p, m) zeros(p);
     zeros(m, p) -JP zeros(m) zeros(m, p) I zeros(m) zeros(m, p) JP zeros(m);
     [zeros(1, p) zeros(1, m) -ℓ zeros(1, p) zeros(1, m) 1.0 zeros(1, p) zeros(1, m) ℓ]]
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
# (J(z)' * J(z)) \ (J(z)' * F(z))
# Δ = zero(z)
# Δ[k] = 1.0
# Δ[2k] = 1.0
# Δ[3k] = 1.0
# gmres!(Δ, J(z), -1.0 * F(z), abstol = 1.0)

function solve()
    ũ = zeros(k)
    u = zeros(k)
    v = zeros(k)
    ũ[end] = 1.0
    u[end] = 1.0
    v[end] = 1.0

    z = [ũ; u; v]
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
                x = z[k .+ (1:p)]
                y = z[k + p .+ (1:m)]
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

        println("iter ($i) - norm: $(norm(F(z)))")

        z .-= α * Δ
    end

    # return z
    x = z[k .+ (1:p)]
    y = z[k + p .+ (1:m)]
    τ = z[k + k]
    κ = z[3k]
    println("τ = $τ")
    println("κ = $κ")
    return x ./ τ, z
end

x_sol, z_sol = solve()

x_sol
b - A * x_sol
norm(x_sol - β.value)

function Fθ(z, θ)
    A = reshape(θ[1:m * p], m, p)
    b = θ[m * p .+ (1:m)]
    c = θ[m * p + m .+ (1:p)]

    Q = Array([zeros(eltype(θ), p, p) A' c;
               -A zeros(eltype(θ), m, m) b;
               -c' -b'      0.0])

    ũ = z[1:k]
    u = z[k .+ (1:k)]
    v = z[2 * k .+ (1:k)]

    [(I + Q) * ũ - (u + v);
     u - Pc(ũ - v);
     ũ - u]
end
θ[m * p + m]
θ = [vec(A); b; c]
_F(y) = Fθ(z_sol, y)
_F(θ)
ForwardDiff.jacobian(_F, θ)
rank(J(z_sol))
J(z_sol)
((J(z_sol)' * J(z_sol)) \ (J(z_sol)' * ForwardDiff.jacobian(_F, θ)))[k .+ (1:p), m * p + (m-1) .+ (1:1+p)] ./ z_sol[2k]
# (J(z_sol) \ ForwardDiff.jacobian(_F, θ))[k .+ (1:p), m * p + (m-1) .+ (1:1+p)] ./ z_sol[2k]
# minimum(ForwardDiff.jacobian(_F, θ)[:, m * p + (m-1) .+ (1:1+p)])
#
# maximum(J(z_sol))
# minimum(J(z_sol))
#
# pinv(J(z_sol))
# F(z_sol)
