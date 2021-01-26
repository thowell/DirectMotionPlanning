using LinearAlgebra, ForwardDiff, StaticArrays
using IterativeSolvers
using Plots

"""
    particle dynamics
    - 3D particle subject to contact forces

    - configuration: q = (x, y, z) ∈ R³
    - impact force (magnitude): n ∈ R₊
    - friction force: β ∈ R⁴₊
        - contact force: λ = (β, n) ∈ R⁴₊ × R₊
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
    # 1-step optimization problem:
    #     find z
    #     s.t. r(z) = 0
    #
    # z = (q, n, b, ψ, η, s)
    #     ψ, η are dual variables for friction subproblem
    #     s are slack variables for convenience

    # initialize
    z = 1.0e-1 * ones(15)
    z[1:3] = copy(q2)

    # if z[3] < 0.1
    #     z[3] = 0.1
    # end

    ρ = 1.0 # barrier parameter
    flag = false

    for k = 1:5
        function r(z)
            # system variables
            q3 = view(z, 1:3)
            n = z[4]
            b = view(z, 5:8)

            λ = [(b[1] - b[3]); (b[2] - b[4]); n] # contact forces
            ϕ = signed_distance(model, q3)        # signed-distance function
            vT = (view(q3, 1:2) - view(q2, 1:2)) ./ h

            # friction subproblem variables
            ψ = z[9]
            η = view(z, 10:13)

            # slack
            s1 = z[14]
            s2 = z[15]

            # action optimality conditions
            [dynamics(model, q1, q2, q3, λ, h);
             s1 - ϕ;
             n * s1 - ρ;

             # maximum dissipation optimality conditions
             [vT; -vT] + ψ * ones(4) - η;
             s2 - (model.μ * n - sum(b));
             ψ * s2 - ρ;
             b .* η .- ρ]
        end

        # Jacobian
        function R(z)
            # differentiate r
            _R = ForwardDiff.jacobian(r, z)

            return _R
        end

        function check_variables(z)
            # q3 = view(z, 1:3)
            n = z[4]
            b = view(z, 5:8)

            # friction subproblem variables
            ψ = z[9]
            η = view(z, 10:13)
            s1 = z[14]
            s2 = z[15]

            if n <= 0.0
                return true
            end

            if any(b .<= 0.0)
                return true
            end

            if any(η .<= 0.0)
                return true
            end

            if ψ <= 0.0
                return true
            end

            if s1 <= 0.0
                return true
            end

            if s2 <= 0.0
                return true
            end

            return false
        end

        extra_iters = 0

        for i = 1:max_iter
            # compute residual, residual Jacobian
            res = r(z)

            if norm(res) < tol
                println("   iter ($i) - norm: $(norm(res))")
                # return z, true
                flag = true
                continue
            end

            jac = R(z)

            # compute step
            # if step_type == :gmres
            #     Δ = gmres(jac, res, abstol = 1.0e-12, maxiter = i + extra_iters)
            # else
            # Δ = (jac' * jac + 0.0e-6 * I) \ (jac' * res) # damped least-squares direction
            Δ = jac \ res
            # end

            # line search the step direction
            α = 1.0

            iter = 0
            while check_variables(z - α * Δ) # backtrack inequalities
                α = 0.5 * α
                # println("   α = $α")
                iter += 1
                if iter > 50
                    @error "backtracking line search fail"
                    flag = false
                    @show n = (z - α * Δ)[4]
                    @show b = view(z - α * Δ, 5:8)
                    @show ψ = (z - α * Δ)[9]
                    @show η = view(z - α * Δ, 10:13)
                    @show s = (z - α * Δ)[14:15]
                    return z, false
                end
            end

            while norm(r(z - α * Δ))^2.0 >= (1.0 - 0.001 * α) * norm(res)^2.0
                α = 0.5 * α
                # println("   α = $α")
                iter += 1
                if iter > 50
                    @error "line search fail"
                    flag = false
                    return z, false
                end
            end

            # update
            z .-= α * Δ
        end

        ρ = 0.1 * ρ
        println("ρ: $ρ")
    end

    return z, flag
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
    b = [zeros(4)]

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
            push!(b, view(z_sol, 5:8))
        end
    end

    return q, n, b
end

# simulation setup
model = Particle(1.0, 9.81, 1.0, 3)
h = 0.01

# initial conditions
# v1 = [1.0; 1.0; 0.0]
# q1 = [0.0; 0.0; 1.0]

v1 = [1.0; 10.0; 0.0]
q1 = [0.0; 0.0; 1.0]

v2 = v1 - gravity(model, q1) * h
q2 = q1 + 0.5 * (v1 + v2) * h

q_sol, y_sol, b_sol = simulate(q1, q2, 1000, h)

plot(hcat(q_sol...)[3:3, :]', xlabel = "", label = "z")
plot!(h .* hcat(y_sol...)', xlabel = "", label = "n", linetype = :steppost)

plot(hcat(q_sol...)[1:2, :]', xlabel = "", label = ["x" "y"])
plot!(h * hcat(b_sol...)', label = ["b1" "b2" "b3" "b4"], linetype = :steppost)
