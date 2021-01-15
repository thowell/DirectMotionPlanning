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
p = 1
n = p + 1

c = (100.0 + 1.0e-8) * ones(n)
G1 = Diagonal(ones(n))
h = zeros(n)
q = zeros(n)
y = (0.0 + 1.0e-8)

A = [-G1; -q']
b = [h; y]
m = size(A, 1)

"Convex.jl"
x = Variable(n)
prob = minimize(c' * x)
prob.constraints += norm(G1 * x + h) <= q' * x + y

@time solve!(prob, ECOS.Optimizer)

@show prob.status
@show prob.constraints
@show x.value
k = n + m + 1

Q = Array([Diagonal(zeros(n)) A' c;
           -A zeros(m, m) b;
           -c' -b'      0.0])

r = m * n + m + n

function Pc(z)
    z_proj = zero(z)
    z_proj[1:n] = z[1:n]
    z_proj[n .+ (1:p+2)] = κ_soc(z[n .+ (1:p+2)])
    # z_proj[n + p + 2 + 1] = κ_no(z[n + p + 2 + 1])
    # z_proj[n + p + 2 + 2] = κ_no(z[n + p + 2 + 2])
    z_proj[n + m + 1] = max(0.0, z[n + m + 1])

    return z_proj
end

function Jc(z)
    J_proj = zeros(k, k)
    J_proj[1:n, 1:n] = Diagonal(ones(n))
    J_proj[n .+ (1:p+2), n .+ (1:p+2)] = Jκ_soc(z[n .+ (1:p+2)])
    # z_proj[n + p + 2 + 1] = κ_no(z[n + p + 2 + 1])
    # z_proj[n + p + 2 + 2] = κ_no(z[n + p + 2 + 2])
    J_proj[n + m + 1, n + m + 1] = z[n + m + 1] >= 0.0 ? 1.0 : 0.0

    return J_proj
end
function Pc_star(z)
    z_proj = zero(z)
    # z_proj[1:n] = z[1:n]
    z_proj[n .+ (1:p+2)] = κ_soc(z[n .+ (1:p+2)])
    # z_proj[n + p + 2 + 1] = κ_no(z[n + p + 2 + 1])
    # z_proj[n + p + 2 + 2] = κ_no(z[n + p + 2 + 2])
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

function Fθ(z, θ)
    A = reshape(θ[1:m * n], m, n)
    b = θ[m * n .+ (1:m)]
    c = θ[m * n + m .+ (1:n)]

    Q = Array([zeros(eltype(θ), n, n) A' c;
               -A zeros(eltype(θ), m, m) b;
               -c' -b'      0.0])

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
    # JP[p+2 .+ (1:1), p+2 .+ (1:1)] = Jκ_no(ũ[n .+ (1:m)][p+2 .+ (1:1)] - v[n .+ (1:m)][p+2 .+ (1:1)])
    # JP[p+2 + 1 .+ (1:1), p+2 + 1 .+ (1:1)] = Jκ_no(ũ[p+2 .+ (1:m)][p+2 + 1 .+ (1:1)] - v[p+2 .+ (1:m)][p+2 + 1 .+ (1:1)])

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
                return x ./ τ, z_sol
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
# x_sol
# x.value
# F(z_sol)
# J(z_sol)

θ = [vec(A); b; c]

u_sol = z_sol[k .+ (1:k)]
v_sol = z_sol[2k .+ (1:k)]

function F_sol(z, θ)
    A = reshape(θ[1:m * n], m, n)
    b = θ[m * n .+ (1:m)]
    c = θ[m * n + m .+ (1:n)]

    Q = Array([zeros(eltype(θ), n, n) A' c;
               -A zeros(eltype(θ), m, m) b;
               -c' -b'      0.0])

   u = z[1:k]
   v = z[k .+ (1:k)]

   [Q * u - v;
    u - Pc(u - v)]
end

w_sol = [u_sol; v_sol]

F_sol(w_sol, θ)
_Fz(y) = F_sol(y, θ)
_Fθ(y) = F_sol(w_sol, y)

Jz = ForwardDiff.jacobian(_Fz, w_sol)
Jθ = ForwardDiff.jacobian(_Fθ, θ)
rank(Jz)
# (Jz \ Jθ)[1:n, m * n + (m-1) .+ (1:1+n)]

((Jz' * Jz + 1.0e-5 * I) \ (Jz' * Jθ))[1:n, m * n + (m-1) .+ (1:1+n)]

norm(z_sol[1:k] - Pc(z_sol[(1:k)]))
norm(Q * z_sol[1:k] - z_sol[2k .+ (1:k)])
norm(z_sol[2k .+ (1:k)] - Pc_star(z_sol[2k .+ (1:k)]))

θ
Rmap(a, Q) = (Q - I) * Pc(a) + a
function _Rmap(a, θ)
    A = reshape(θ[1:m * n], m, n)
    b = θ[m * n .+ (1:m)]
    c = θ[m * n + m .+ (1:n)]

    Q = Array([zeros(eltype(θ), n, n) A' c;
               -A zeros(eltype(θ), m, m) b;
               -c' -b'      0.0])

    Rmap(a, Q)
end
function Rzmap(a, Q)
    (Q - I) * Jc(a) + I
end
function _Rzmap(a, θ)
    A = reshape(θ[1:m * n], m, n)
    b = θ[m * n .+ (1:m)]
    c = θ[m * n + m .+ (1:n)]

    Q = Array([zeros(eltype(θ), n, n) A' c;
               -A zeros(eltype(θ), m, m) b;
               -c' -b'      0.0])

    Rzmap(a, Q)
end
y_sol = z_sol[1:k] - z_sol[2k .+ (1:k)]
_Rmap(y_sol, θ)
tmp(w) = _Rmap(y_sol, w)
_Rzmap(y_sol, θ)

((_Rzmap(y_sol, θ)' * _Rzmap(y_sol, θ) + 1.0e-5 * I) \ (_Rzmap(y_sol, θ)' * ForwardDiff.jacobian(tmp, θ)))[1:n, m * n + (m-1) .+ (1:1+n)]
