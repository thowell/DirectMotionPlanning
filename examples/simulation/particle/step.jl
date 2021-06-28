"""
    step particle system
        solves 1-step feasibility problem
"""
function step(q1, q2, u1, h;
    tol = 1.0e-8, max_iter = 100)
    # 1-step optimization problem:
    #     find z
    #     s.t. r(z) = 0
    #
    # z = (q, n, sϕ, sb, db)
    #     s are slack variables for convenience
    #     sb[2:3] is the friction force

    # initialize
    z = 1.0 * ones(11)
    z[1:3] = copy(q2)

    # initialize soc variables
    sb, _ = κ_so(z[6:8])
    sb[1] += 1.0
    db, _ = κ_so(z[9:11])
    db[1] += 1.0

    z[6:8] = sb
    z[9:11] = db

    μ = 1.0 # barrier parameter
    flag = false

    θ = [q1; q2; u1]

    e = [1.0; 0.0; 0.0]

    function r(z, θ)
        # system variables
        q3 = view(z, 1:3)
        n = z[4]
        sϕ = z[5]
        sb = view(z, 6:8)
        db = view(z, 9:11)

        q1 = view(θ, 1:3)
        q2 = view(θ, 4:6)
        u1 = view(θ, 7:9)

        λ = [view(sb, 2:3); n] # contact forces
        ϕ = signed_distance(model, q3) # signed-distance function
        vT = (view(q3, 1:2) - view(q2, 1:2)) ./ h

        # action optimality conditions
        [dynamics(model, q1, q2, q3, u1, λ, h);
         sϕ - ϕ;
         n * sϕ - μ;

         # maximum dissipation optimality conditions
         vT - view(db, 2:3);
         sb[1] - model.friction_coeff * n;
         cone_product(db, sb) - μ * e]
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

    function check_variables(z)
        # system variables
        q3 = view(z, 1:3)
        n = z[4]
        sϕ = z[5]
        sb = view(z, 6:8)
        db = view(z, 9:11)

        if n <= 0.0
            return true
        end

        if sϕ <= 0.0
            return true
        end

        if !κ_so(sb)[2]
            # println("bs not in cone")
            return true
        end

        if !κ_so(db)[2]
            # println("bz not in cone")
            return true
        end

        return false
    end

    for k = 1:4
        extra_iters = 0
        for i = 1:max_iter
            # compute residual, residual Jacobian
            res = r(z, θ)
            if norm(res) < tol
                # println("   iter ($i) - norm: $(norm(res))")
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
                    flag = false
                    return z, false
                end
            end

            while norm(r(z - α * Δ, θ))^2.0 >= (1.0 - 0.001 * α) * norm(res)^2.0
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

        μ = 0.1 * μ
        # println("μ: $μ")
    end

    # Δz = -1.0 * (Rz(z, θ)' * Rz(z, θ) + 1.0e-6 * I) \ (Rz(z, θ)' * Rθ(z, θ)) # damped least-squares direction
    Δz = -1.0 * Rz(z, θ) \ Rθ(z, θ)

    q3 = view(z, 1:3)
    n = z[4]
    b = view(z, 6:8)
    Δq1 = view(Δz, 1:3, 1:3)
    Δq2 = view(Δz, 1:3, 3 .+ (1:3))
    Δu1 = view(Δz, 1:3, 6 .+ (1:3))

    return q3, n, b, Δq1, Δq2, Δu1, flag
end

"""
    step particle system
        solves 1-step feasibility problem
"""
function step_slack(q1, q2, u1, h;
    tol = 1.0e-8, max_iter = 100)
    # 1-step optimization problem:
    #     find z
    #     s.t. r(z) = 0
    #
    # z = (q, n, sϕ, sb, db)
    #     s are slack variables for convenience
    #     sb[2:3] is the friction force

    # initialize
    z = 1.0 * ones(11)
    z[1:3] = copy(q2)

    # initialize soc variables
    sb, _ = κ_so(z[6:8])
    sb[1] += 1.0
    db, _ = κ_so(z[9:11])
    db[1] += 1.0

    z[6:8] = sb
    z[9:11] = db

    # μ = 1.0 # barrier parameter
    flag = false

    θ = [q1; q2; u1]

    e = [1.0; 0.0; 0.0]

    function r(z, θ)
        # system variables
        q3 = view(z, 1:3)
        n = z[4]
        sϕ = z[5]
        sb = view(z, 6:8)
        db = view(z, 9:11)

        q1 = view(θ, 1:3)
        q2 = view(θ, 4:6)
        u1 = view(θ, 7:9)
        μ = θ[10]

        λ = [view(sb, 2:3); n] # contact forces
        ϕ = signed_distance(model, q3) # signed-distance function
        vT = (view(q3, 1:2) - view(q2, 1:2)) ./ h

        # @show u1
        # action optimality conditions
        [dynamics(model, q1, q2, q3, u1, λ, h);
         sϕ - ϕ;
         n * sϕ - μ;

         # maximum dissipation optimality conditions
         vT - view(db, 2:3);
         sb[1] - model.friction_coeff * n;
         cone_product(db, sb) - μ * e]
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

    function check_variables(z)
        # system variables
        q3 = view(z, 1:3)
        n = z[4]
        sϕ = z[5]
        sb = view(z, 6:8)
        db = view(z, 9:11)

        if n <= 0.0
            return true
        end

        if sϕ <= 0.0
            return true
        end

        if !κ_so(sb)[2]
            # println("bs not in cone")
            return true
        end

        if !κ_so(db)[2]
            # println("bz not in cone")
            return true
        end

        return false
    end

    for i = 1:max_iter
        # compute residual, residual Jacobian
        res = r(z, θ)
        if norm(res) < tol
            # println("   iter ($i) - norm: $(norm(res))")
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
                flag = false
                return z, false
            end
        end

        while norm(r(z - α * Δ, θ))^2.0 >= (1.0 - 0.001 * α) * norm(res)^2.0
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

    # Δz = -1.0 * (Rz(z, θ)' * Rz(z, θ) + 1.0e-6 * I) \ (Rz(z, θ)' * Rθ(z, θ)) # damped least-squares direction
    Δz = -1.0 * Rz(z, θ) \ Rθ(z, θ)

    q3 = view(z, 1:3)
    n = z[4]
    b = view(z, 6:8)
    Δq1 = view(Δz, 1:3, 1:3)
    Δq2 = view(Δz, 1:3, 3 .+ (1:3))
    Δu1 = view(Δz, 1:3, 6 .+ (1:4))

    return q3, n, b, Δq1, Δq2, Δu1, flag
end
