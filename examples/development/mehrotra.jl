using LinearAlgebra, ForwardDiff

# test Mehrotra predictor-corrector algorithm
n = 10
me = 5
mi = n
p = 0

idx_ineq = collect(1:mi)


x0 = rand(n)
p0 = zeros(p)
a0 = zeros(mi)

# objective
c = rand(n)
function obj(x)
    transpose(c) * x
end

# constraints
A = rand(me, n)
b = A * x0
function con(x)
    A * x - b
end

# unpack
function unpack(w)
    x = view(w, 1:n)
    y = view(w, n .+ (1:me))
    z = view(w, n + me .+ (1:mi))

    return x, y, z
end

# barrier check
function check_variables(w)
    x, y, z = unpack(w)
    if any(x .<= 0.0)
        return true
    end
    if any(z .<= 0.0)
        return true
    end
    return false
end


function r(w, θ, a)
    x, y, z = unpack(w)

    lag = c + transpose(A) * y
    lag[idx_ineq] -= z

    c_eq = con(x)

    comp = x[idx_ineq] .* z
    comp -= a

    return [lag; c_eq; comp]
end

function Rz(w, θ)
    return ForwardDiff.jacobian(q -> r(q, θ, a0), w)
end

function Rθ(w, θ)
    return ForwardDiff.jacobian(q -> r(w, q, a0), θ)
end

function interior_point(x0, θ;
    x_init_tol = 0.1,
    r_tol = 1.0e-8,
    μ_tol = 1.0e-5,
    max_iter_inner = 100,
    max_iter_ls = 25)

    # initialization
    x = max.(x0, x_init_tol)
    y = zeros(me)
    z = ones(mi)
    w = [x; y; z]
    μ = 1.0
    e = ones(mi)

    flag = false
    total_iter = 0

    while μ > μ_tol
        for i = 1:max_iter_inner

            # compute residual, residual Jacobian
            res = r(w, θ, μ * e)
            res_norm = norm(res)

            # println("iteration: $i, residual: $res_norm")

            if res_norm < r_tol
                flag = true
                continue
            end

            jac = Rz(w, θ)

            # compute step
            Δ = -jac \ res

            # line search the step direction
            α = 1.0

            iter = 0
            while check_variables(w + α * Δ) # backtrack inequalities
                α = 0.5 * α
                # println("   α = $α")
                iter += 1
                if iter > max_iter_ls
                    @error "backtracking line search fail"
                    flag = false
                    return w, zeros(n + me + mi, p), false, total_iter
                end
            end

            while norm(r(w + α * Δ, θ, μ * e))^2.0 >= (1.0 - 0.001 * α) * res_norm^2.0
                α = 0.5 * α
                # println("   α = $α")
                iter += 1
                if iter > max_iter_ls
                    @error "line search fail"
                    flag = false
                    return w, zeros(n + me + mi, p), false, total_iter
                end
            end

            # update
            w .+= α * Δ

            total_iter += 1
        end

        μ = 0.1 * μ
        println("μ: $μ")
    end

    δw = -1.0 * Rz(w, θ) \ Rθ(w, θ)

    return w, δw, flag, total_iter
end

w_soli, δw_soli, flag, total_iter = interior_point(rand(n), p0)
@show total_iter

function mehrotra(x0, θ;
    x_init_tol = 0.1,
    r_tol = 1.0e-8,
    μ_tol = 1.0e-5,
    max_iter_inner = 100,
    max_iter_ls = 25)

    # initialization
    x = max.(x0, x_init_tol)
    y = zeros(me)
    z = ones(mi)
    w = [x; y; z]
    μ = 1.0
    e = ones(mi)

    flag = false
    total_iter = 0

    while μ > μ_tol
        for i = 1:max_iter_inner

            # unpack
            x, y, z = unpack(w)

            # compute residual, residual Jacobian
            res_aff = r(w, θ, a0)
            res_aff_norm = norm(res_aff, 1)

            # println("iteration: $i, residual: $res_norm")

            if res_aff_norm < r_tol
                flag = true
                continue
            end

            jac = Rz(w, θ)

            # compute affine step
            Δ_aff = -jac \ res_aff

            Δx_aff, Δy_aff, Δz_aff = unpack(Δ_aff)

            α_aff_pr = min(1.0, minimum([Δxi < 0.0 ? -x[i] / Δxi : Inf for (i, Δxi) in enumerate(Δx_aff)]))
            α_aff_du = min(1.0, minimum([Δzi < 0.0 ? -z[i] / Δzi : Inf for (i, Δzi) in enumerate(Δz_aff)]))
            μ_aff = transpose(x[idx_ineq] + α_aff_pr * Δx_aff[idx_ineq]) * (z + α_aff_du * Δz_aff) / mi

            # compute corrector step
            σ = (μ_aff / μ)^3.0

            res_corr = r(w, θ, -Δx_aff[idx_ineq] .* Δz_aff + σ * μ * e)

            Δ_corr = -jac \ res_corr

            # line search the step direction
            η = 1.0
            α_pr = α_aff_pr
            α_du = α_aff_du

            α = η .* [α_pr * ones(n); α_du * ones(me + mi)]

            iter = 0
            # while check_variables(w + α .* Δ_corr) # backtrack inequalities
            #     η = 0.99 * η
            #     α = η .* [α_pr * ones(n); α_du * ones(me + mi)]
            #
            #     # println("   α = $α")
            #     iter += 1
            #     if iter > max_iter_ls
            #         @error "backtracking line search fail"
            #         flag = false
            #         return w, zeros(n + me + mi, p), false, total_iter
            #     end
            # end

            # while norm(r(w + α .* Δ_corr, θ, a0))^2.0 >= (1.0 - 0.001 * η) * res_aff_norm^2.0
            #     η = 0.99 * η
            #     α = η .* [α_pr * ones(n); α_du * ones(me + mi)]
            #     # println("   α = $α")
            #     iter += 1
            #     if iter > max_iter_ls
            #         @error "line search fail"
            #         flag = false
            #         return w, zeros(n + me + mi, p), false, total_iter
            #     end
            # end

            # update
            w .+= α .* Δ_corr

            total_iter += 1
        end

        μ = 0.1 * μ
        println("μ: $μ")
    end

    δw = -1.0 * Rz(w, θ) \ Rθ(w, θ)

    return w, δw, flag, total_iter
end

w_solm, δw_solm, flag, total_iter = mehrotra(x0, p0)
@show total_iter

obj(w_soli[1:n])
obj(w_solm[1:n])

con(w_soli[1:n])
con(w_solm[1:n])
