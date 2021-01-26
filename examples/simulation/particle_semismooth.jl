using LinearAlgebra, ForwardDiff, StaticArrays
using IterativeSolvers
using Plots

"""
    cones
    - nonnegative orthant (no)
        x >= 0

    - second-order (so)
        ||z1|| <= z2

    A Semismooth Newton Method for Fast, Generic Convex Programming
        https://arxiv.org/abs/1705.00772

    # todo: try backtracking linesearch for cone
        https://digital.lib.washington.edu/researchworks/bitstream/handle/1773/43279/Dueri_washington_0250E_19426.pdf?isAllowed=y&sequence=1
"""

# nonnegative orthant cone
function κ_no(z)
    max.(0.0, z)
end

# nonnegative orthant cone Jacobian
function Jκ_no(z)
    p = zero(z)
    for (i, pp) in enumerate(z)
        if pp >= 0.0
            p[i] = 1.0
        end
    end
    return Diagonal(p)
end

# second-order cone
function κ_so(z)
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

# second-order cone Jacobian
function Jκ_so(z)
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

"""
    particle dynamics
    - 3D particle subject to contact forces

    - configuration: q = (x, y, z) ∈ R³
    - impact force (magnitude): n ∈ R₊
    - friction force: b ∈ R²
        - contact force: λ = (b, n) ∈ R² × R₊
        - friction coefficient: μ ∈ R₊

    Discrete Mechanics and Variational Integrators
        pg. 363
"""
struct Particle
      m # mass
      g # gravity
      μ # friction coefficient

      nq # configuration dimension
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

# signed distance function
function signed_distance(model, q)
    q[3]
end

# contact force Jacobian
function jacobian(model, q)
    Diagonal(@SVector ones(model.nq))
end

# dynamics
function dynamics(model, q1, q2, q3, λ, h)
    nq = model.nq
    SVector{nq}(mass_matrix(model) * (2.0 * q2 - q1 - q3) / h
        - h * gravity(model, q2)
        + h * jacobian(model, q3)' * λ)
end

"""
    step particle system
        solves 1-step feasibility problem
"""
function _step(q1, q2, h;
    tol = 1.0e-8, z0_scale = 0.001, step_type = :gmres, max_iter = 100)
    """
        1-step optimization problem:
            find z
            s.t. r(z) = 0

        z = (q, n, b, ν, η)
            ν, η are slack variables for friction subproblem
    """
    function r(z)
        # system variables
        q3 = view(z, 1:3)
        n = z[4]
        b = view(z, 5:6)

        λ = [b; n]                     # contact forces
        ϕ = signed_distance(model, q3) # signed-distance function

        # friction subproblem variables
        b̄ = view(z, 5:7)
        ν = z[8]
        η = view(z, 9:11)

        # action optimality conditions
        [dynamics(model, q1, q2, q3, λ, h);
         ϕ - κ_no(ϕ - n);

         # maximum dissipation optimality conditions
         [(view(q3, 1:2) - view(q2, 1:2)) ./ h; -ν] - η;
          model.μ * n - b̄[end];
          b̄ - κ_so(b̄ - η)]
    end

    # Jacobian
    function R(z)
        # differentiate r
        _R = ForwardDiff.jacobian(r, z)

        # correct projection derivatives
        q3 = view(z, 1:3)
        ϕ = signed_distance(model, q3) # signed-distance function
        n = z[4]
        b̄ = view(z, 5:7)
        ν = z[8]
        η = view(z, 9:11)

        J_no = Jκ_no([ϕ - n])
        J_so = Jκ_so(b̄ - η)

        _R[4, 3] = 1.0 - J_no[1]
        _R[4, 4] = J_no[1]
        _R[(end-2):end, 5:11] = [(Diagonal(ones(3)) - J_so) zeros(3, 1) J_so]

        return _R
    end

    # initialize
    z = z0_scale * rand(11)
    z[1:3] = copy(q2)

    extra_iters = 0

    for i = 1:max_iter
        # compute residual, residual Jacobian
        res = r(z)

        if norm(res) < tol
            println("   iter ($i) - norm: $(norm(r(z)))")
            return z, true
        end

        jac = R(z)

        # compute step
        if step_type == :gmres
            Δ = gmres(jac, res, abstol = 1.0e-12, maxiter = i + extra_iters)
        else
            Δ = (jac' * jac + 1.0e-6 * I) \ (jac' * res) # damped least-squares direction
        end

        # line search the step direction
        iter = 0
        α = 1.0
        while norm(r(z - α * Δ))^2.0 >= (1.0 - 0.001 * α) * norm(res)^2.0 && α > 1.0e-4
            α = 0.5 * α
            # println("   α = $α")
            iter += 1
            if iter > 100
                @error "line search fail"

                return z, false
            end
        end

        if α <= 1.0e-4
            extra_iters += 1
        end

        println("   iter ($i) - norm: $(norm(r(z)))")

        # update
        z .-= α * Δ
    end

    return z, false
end

"""
    simulate
    - solves 1-step feasibility problem for T time steps
    - initial configurations: q1, q2 (note this can encode initial velocity)
    - time step: h
"""
function simulate(q1, q2, T, h;
    z0_scale = 0.0, step_type = :gmres)
    println("simulation")

    # initialize histories
    q = [q1, q2]
    n = [0.0]
    b = [zeros(2)]

    # step
    for t = 1:T
        println("   t = $t")
        z_sol, status = _step(q[end-1], q[end], h,
            z0_scale = z0_scale, step_type = step_type)

        if !status
            @error "failed step (t = $t)"
            return q, n, b
        else
            push!(q, view(z_sol, 1:3))
            push!(n, z_sol[4])
            push!(b, view(z_sol, 5:6))
        end
    end

    return q, n, b
end

# simulation setup
model = Particle(1.0, 9.81, 1.0, 3)
h = 0.05

# initial conditions
# v1 = [1.0; 1.0; 0.0]
# q1 = [0.0; 0.0; 1.0]

v1 = [-1.0; 10.0; 0.0]
q1 = [0.0; 0.0; 1.0]

v2 = v1 - gravity(model, q1) * h
q2 = q1 + 0.5 * (v1 + v2) * h

q_sol, y_sol, b_sol = simulate(q1, q2, 500, h,
    z0_scale = 0.0, step_type = :gmres)

plot(hcat(q_sol...)[3:3, :]', xlabel = "", label = "z")
plot!(h .* hcat(y_sol...)', xlabel = "", label = "n", linetype = :steppost)

plot(hcat(q_sol...)[1:2, :]', xlabel = "", label = ["x" "y"])
plot!(h * hcat(b_sol...)', label = ["b1" "b2"], linetype = :steppost)
