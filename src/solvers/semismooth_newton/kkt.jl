using LinearAlgebra, ForwardDiff, Distributions, StaticArrays
using Convex, SCS, ECOS
using IterativeSolvers

# noc
function κ_no(z)
    max.(0.0, z)
end

function Jκ_no(z)
    p = zero(z)
    for (i, pp) in enumerate(z)
        println(pp)
        println(i)
        if pp >= 0.0
            p[i] = 1.0
        end
    end
    return Diagonal(p)
end

# soc
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

# friction problem
v = 1.0 * ones(2)
y = 1.0 #1.0e-3

n = 3
m = 1
c = [v; 0.0]
A = [0.0; 0.0; 1.0]'
b = copy(y)

θ_friction = [vec(A); b; c]

function r_friction(z)
    x = z[1:3]
    μ = z[3 + 1]
    λ = z[3 + 1 .+ (1:3)]

    [[v; 0.0] - [0.0; 0.0; 1.0] * μ - λ;
     y - [0.0; 0.0; 1.0]' * x;
     x - κ_soc(x - λ)]
end

function r_friction(z, θ)
    x = z[1:3]
    μ = z[3 + 1]
    λ = z[3 + 1 .+ (1:3)]

    A = reshape(θ[1:1 * 3], 1, 3)
    b = θ[1 * 3 .+ (1:1)]
    c = θ[1 * 3 + 1 .+ (1:3)]

    [c - A' * μ - λ;
     b - A * x;
     x - κ_soc(x - λ)]
end

function R_friction(z)
    x = z[1:3]
    μ = z[3 + 1]
    λ = z[3 + 1 .+ (1:3)]
    J = Jκ_soc(x - λ)

    [zeros(3, 3) -[0.0; 0.0; 1.0]'' -Diagonal(ones(3));
     -[0.0; 0.0; 1.0]' zeros(1, 1) zeros(1, 3);
     (Diagonal(ones(3)) - J) zeros(3, 1) J]
end

function solve()
    z = 0.001 * rand(3 + 1 + 3)

    extra_iters = 0

    for i = 1:10
        _F = r_friction(z)
        _J = R_friction(z) #R_friction(z)
        Δ = gmres(_J, 1.0 * _F, abstol = 1.0e-12, maxiter = i + extra_iters)
        # Δ = (_J' * _J + 1.0e-5 * I) \ (_J' * _F)
        iter = 0
        α = 1.0
        while norm(r_friction(z - α * Δ))^2.0 >= (1.0 - 0.001 * α) * norm(_F)^2.0 && α > 1.0e-4
            α = 0.5 * α
            # println("   α = $α")
            iter += 1
            if iter > 100
                @error "line search fail"

                return z
            end
        end

        if α <= 1.0e-4
            extra_iters += 1
        end

        println("iter ($i) - norm: $(norm(r_friction(z)))")

        z .-= α * Δ
    end

    return z
end

z_sol = solve()

@show b_sol = z_sol[1:2]


θ = [vec(A); b; c]
r_friction(z_sol, θ)

_rz(w) = r_friction(w, θ)
_rθ(w) = r_friction(z_sol, w)

Rz = ForwardDiff.jacobian(_rz, z_sol)
Rθ = ForwardDiff.jacobian(_rθ, θ)

dθ = ((Rz' * Rz + 1.0e-5 * I) \ (Rz' * Rθ))[1:2, m * n .+ (1:m+(n-1))]

# impact problem
# rod
struct Particle
      m # mass
      g # gravity

      nq # configuration dimension
end

# jacobian
function jacobian(model, q)
    Diagonal(@SVector ones(model.nq))
end

# mass matrix
function mass_matrix(model)
      m = model.m

      Diagonal(@SVector [m, m, m])
end

# gravity
function gravity(model, q)
      m = model.m
      g = model.g

      @SVector [0.0, 0.0, m * g]
end

# dynamics
function dynamics(model, q1, q2, q3, λ, h)
      nq = model.nq
      SVector{nq}(mass_matrix(model) * (2.0 * q2 - q1 - q3) / h
            - h * gravity(model, q2)
            + h * jacobian(model, q3)' * λ)
end


# create model
model = Particle(1.0, 9.81, 3)

q1 = [0.0; 0.0; 1.0]
q2 = [1.0; 1.0; 1.0]

function r_action(z)
    q3 = z[1:3]
    q3z = z[3]
    λ = z[4]

    [dynamics(model, q1, q2, q3, [0.0; 0.0; λ], 0.1);
     q3z - κ_no(q3z - λ)]
end

function solve()
    z = [q2; 0.001]

    extra_iters = 0

    for i = 1:10
        _F = r_action(z)
        _J = ForwardDiff.jacobian(r_action, z)
        Δ = gmres(_J, 1.0 * _F, abstol = 1.0e-12, maxiter = i + extra_iters)
        # Δ = (_J' * _J + 1.0e-5 * I) \ (_J' * _F)
        iter = 0
        α = 1.0
        while norm(r_action(z - α * Δ))^2.0 >= (1.0 - 0.001 * α) * norm(_F)^2.0 && α > 1.0e-4
            α = 0.5 * α
            # println("   α = $α")
            iter += 1
            if iter > 100
                @error "line search fail"

                return z
            end
        end

        if α <= 1.0e-4
            extra_iters += 1
        end

        println("iter ($i) - norm: $(norm(r_action(z)))")

        z .-= α * Δ
    end

    return z
end

z_sol = solve()

function r_friction(z)
    x = z[1:3]
    μ = z[3 + 1]
    λ = z[3 + 1 .+ (1:3)]

    [[v; 0.0] - [0.0; 0.0; 1.0] * μ - λ;
     y - [0.0; 0.0; 1.0]' * x;
     x - κ_soc(x - λ)]
end

# impact and friction
q1 = [1.5; 0.5; 0.1]
q2 = [10.0; 0.75; 0.075]

function r_impact_friction(z)
    q3 = z[1:3]
    q3z = z[3]
    v = (q3[1:2] - q2[1:2]) ./ 0.1
    λ = z[4]
    b = z[5:6]
    b̄ = z[5:7]
    μ = z[8]
    η = z[9:11]

    [dynamics(model, q1, q2, q3, [b; λ], 0.1);
     q3z - κ_no(q3z - λ);

     [v; 0.0] - [0.0; 0.0; 1.0] * μ - η;
      λ - [0.0; 0.0; 1.0]' * b̄;
      b̄ - κ_soc(b̄ - η)]
end

function R_impact_friction(z)
    R = ForwardDiff.jacobian(r_impact_friction, z)

    # fix projection
    b̄ = z[5:7]
    μ = z[8]
    η = z[9:11]
    J = Jκ_soc(b̄ - η)

    R[(end-2):end, 5:11] = [(Diagonal(ones(3)) - J) zeros(3, 1) J]

    return R
end

function solve()
    z = 0.001 * rand(11)
    z[1:3] = copy(q2)

    extra_iters = 0

    for i = 1:25
        _F = r_impact_friction(z)
        _J = R_impact_friction(z)
        Δ = gmres(_J, 1.0 * _F, abstol = 1.0e-12, maxiter = i + extra_iters)
        # Δ = (_J' * _J + 1.0e-5 * I) \ (_J' * _F)
        iter = 0
        α = 1.0
        while norm(r_impact_friction(z - α * Δ))^2.0 >= (1.0 - 0.001 * α) * norm(_F)^2.0 && α > 1.0e-4
            α = 0.5 * α
            # println("   α = $α")
            iter += 1
            if iter > 100
                @error "line search fail"

                return z
            end
        end

        if α <= 1.0e-4
            extra_iters += 1
        end

        println("iter ($i) - norm: $(norm(r_impact_friction(z)))")

        z .-= α * Δ
    end

    return z
end

z_sol = solve()

z_sol[1:3]
z_sol[4]
z_sol[5:6]
