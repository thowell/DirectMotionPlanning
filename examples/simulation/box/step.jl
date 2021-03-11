function check_variables(z)
    q3, n, sϕ, b, sb = unpack(z)

    if any([ni <= 0.0 for ni in n])
        # println("n not in cone")
        return true
    end

    if any([si <= 0.0 for si in sϕ])
        # println("ϕ not in cone")
        return true
    end

    if any([!κ_so(bi)[2] for bi in b])
        # println("b not in cone")
        return true
    end

    if any([!κ_so(si)[2] for si in sb])
        # println("bs not in cone")
        return true
    end

    return false
end

function r(z, θ)
    q3, n, sϕ, b, sb = unpack(z)
    q2 = view(θ, nq .+ (1:nq))
    q1 = view(θ, 1:nq)
    u1 = view(θ, 2 * nq .+ (1:nu))
    h = θ[end-1]
    μ = θ[end]

    λ = vcat([[view(b[i], 2:3); n[i]] for i = 1:num_contacts]...) # contact forces
    ϕ = signed_distance(model, q3) # signed-distance function
    vT = P_func(model, q3) * (q3 - q2) ./ h

    e = [1.0; 0.0; 0.0]

    sb_stack = vcat([view(sb[i], 2:3) for i = 1:num_contacts]...)
    fc_stack = vcat([b[i][1] - model.μ * n[i] for i = 1:num_contacts]...)
    cp_stack = vcat([cone_product(sb[i], b[i]) - μ * e for i = 1:num_contacts]...)

    # action optimality conditions
    [dynamics(model, q1, q2, q3, u1, λ, h);
     sϕ - ϕ;
     n .* sϕ .- μ;

     # maximum dissipation optimality conditions
     vT - sb_stack;
     fc_stack;
     cp_stack]
end

# Jacobian
function Rz(z, θ)
    # differentiate r
    _r(w) = r(w, θ)
    _R = ForwardDiff.jacobian(_r, z)
    return _R
end

function Rθ(z, θ)
    # differentiate r
    _r(w) = r(z, w)
    _R = ForwardDiff.jacobian(_r, θ)

    return _R
end

"""
    step particle system
        solves 1-step feasibility problem
"""
function step(q1, q2, u1, h;
    r_tol = 1.0e-5,
    μ_tol = 1.0e-5,
    z_init = 1.0,
    μ_init = 1.0,
    max_iter = 100)
    # 1-step optimization problem:
    #     find z
    #     s.t. r(z) = 0
    #
    # z = (q, n, sϕ, b, sb)
    #     s are slack variables for convenience
    #     b[2:3] is the friction force

    z = initialize(q2, num_var, z_init = z_init)
    μ = μ_init # barrier parameter
    θ = [q1; q2; u1; h; μ] # problem data

    flag = false

    for k = 1:10
        for i = 1:max_iter
            # compute residual, residual Jacobian
            res = r(z, θ)
            if norm(res) < r_tol
                # println("   iter ($i) - norm: $(norm(res))")
                # println("     μ = $μ")
                # return z, true
                flag = true
                continue
            end

            jac = Rz(z, θ)

            # compute step
            Δ = jac \ res

            # line search the step direction
            α = 1.0

            iter = 0
            while check_variables(z - α * Δ) # backtrack inequalities
                α = 0.5 * α

                # println("   α = $α")
                iter += 1
                if iter > 50
                    @error "backtracking line search fail"
                    q3, n, sϕ, b, sb = unpack(z)
                    println("q3: $q3")
                    println("q2: $q2")
                    println("q1: $q1")
                    println("μ: $μ")
                    flag = false
                    return q2, zeros(num_contacts), zeros(3 * num_contacts), zeros(nq, nq), zeros(nq, nq), zeros(nq , nu), false
                end
            end

            while norm(r(z - α * Δ, θ), Inf)^2.0 >= (1.0 - 0.001 * α) * norm(res, Inf)^2.0
                α = 0.5 * α
                # println("   α = $α")

                iter += 1
                if iter > 50
                    @error "line search fail"
                    q3, n, sϕ, b, sb = unpack(z)
                    println("q3: $q3")
                    println("q2: $q2")
                    println("q1: $q1")
                    println("μ: $μ")

                    flag = false
                    return q2, zeros(num_contacts), zeros(3 * num_contacts), zeros(nq, nq), zeros(nq, nq), zeros(nq , nu), false
                end
            end

            # update
            z .-= α * Δ
        end

        if μ < μ_tol
            break
        else
            μ = 0.1 * μ
            θ[end] = μ
        end
    end

    Δz = -1.0 * Rz(z, θ) \ Rθ(z, θ)

    q3, n, sϕ, b, sb = unpack(z)
    Δq1 = view(Δz, 1:nq, 1:nq)
    Δq2 = view(Δz, 1:nq, nq .+ (1:nq))
    Δu1 = view(Δz, 1:nq, 2 * nq .+ (1:nu))

    return q3, n, b, Δq1, Δq2, Δu1, flag
end
