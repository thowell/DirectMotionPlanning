using LinearAlgebra, ForwardDiff, StaticArrays
using IterativeSolvers
using Plots

"""
    second-order cone
"""
function κ_so(z)
    z1 = z[2:end]
    z0 = z[1]

    z_proj = zero(z)
    status = false

    if norm(z1) <= z0
        z_proj = copy(z)
        if norm(z1) < z0
            status = true
        end
        # status = true
    elseif norm(z1) <= -z0
        z_proj = zero(z)
    else
        a = 0.5 * (1.0 + z0 / norm(z1))
        z_proj[1:end-1] = a * z1
        z_proj[end] = a * norm(z1)
    end

    return z_proj, status
end
J = [1.0 zeros(1, 2);
     zeros(2) Diagonal(-1.0 * ones(2))]

# H_inv(u) = 2.0 * u * u' - (u' * J * u) * J
# H_inv_sq(u) = [u[1] u[2:3]';
#                u[2:3] (u[2:3] * u[2:3]') / (u[1] + (u' * J * u)^0.5) + ((u' * J * u)^0.5) * Diagonal(ones(2))]
# H_sq(u) = (1.0 / (u' * J * u)) * [u[1] -u[2:3]'; -u[2:3] ((u[1] + (u' * J * u)^0.5) \ (u[2:3] * u[2:3]') + ((u' * J * u)^0.5) * I)]
# barrier_grad(u) = -1.0 * (u' * J * u) \ (J * u)
#
# ss = [2.0; 0.5; 0.5]
# zz = -1.0 * barrier_grad(ss)
# H_sq(ss) * H_inv_sq(ss)

function normalized_vector(u)
    (1.0 / (u' * J * u)^0.5) * u
end
# norm(normalized_vector(qq))

function gamma(z, s)
    (0.5 * (1.0 + normalized_vector(z)' * normalized_vector(s)))^0.5
end

function w̄(z, s)
    0.5 * gamma(z, s) * (normalized_vector(s) + J * normalized_vector(z))
end

# H_inv(w̄(zz, ss)) * normalized_vector(zz) - normalized_vector(ss)

function W̄(z, s)
    w = w̄(z, s)
    [w[1] w[2:3]'; w[2:3] (I + (w[1] + 1.0)) \ (w[2:3] * w[2:3]')]
end
# W̄(ss, qq)

function W̄inv(z, s)
    w = w̄(z, s)
    [w[1] -w[2:3]'; -w[2:3] (I + (w[1] + 1.0)) \ (w[2:3] * w[2:3]')]
end

function cone_product(z, s)
    [z' * s; z[1] * s[2:3] + s[1] * z[2:3]]
end

function W(z, s)
    _W̄ = W̄(z, s)
    (((s' * J * s) / (z' * J * z))^0.25) * _W̄
end

# inv(W(zz,ss))

# ww, _ = κ_so(rand(3))
# ww[1] += 1.0
# hh = H_inv(ww)
# hhsq = H_inv_sq(ww)
# ss = sqrt(hh)
# ss' * ss

# @show x = rand(3)
# @show x_proj, status = κ_so(x)
# x_proj[1] += 1.0
# x_proj
# @show x_proj_proj, status = κ_so(x_proj)

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
    # println(q1)
    # println(q2)
    # println(q3)
    # println(λ)
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
    # z = (q, n, b, by, bz, s)
    #     s are slack variables for convenience

    # initialize
    z = 1.0e-1 * ones(13)
    z[1:3] = copy(q2)

    # initialize soc variables
    bs, _ = κ_so(z[7:9])
    bs[1] += 1.0
    bz, _ = κ_so(z[10:12])
    bz[1] += 1.0

    z[7:9] = bs
    z[10:12] = bz

    ρ = 1.0 # barrier parameter
    flag = false

    for k = 1:5
        function r(z)
            # system variables
            q3 = view(z, 1:3)
            n = z[4]
            b = view(z, 5:6)
            bs = view(z, 7:9)
            bz = view(z, 10:12)
            ϕs = z[13]

            λ = [b; n] # contact forces
            ϕ = signed_distance(model, q3) # signed-distance function
            vT = (view(q3, 1:2) - view(q2, 1:2)) ./ h

            G = [zeros(1, 2); -Diagonal(ones(2))]
            g = [model.μ * n; zeros(2)]

            # W = W̄(bz, bs)
            # Winv = W̄inv(bz, bs)

            # action optimality conditions
            [dynamics(model, q1, q2, q3, λ, h);
             ϕs - ϕ;
             n * ϕs - ρ;

             # maximum dissipation optimality conditions
             G' * bz + vT;
             bs + G * b - g;
             cone_product(bz, bs) - ρ * [1.0; 0.0; 0.0]]
        end

        # Jacobian
        function R(z)
            # differentiate r
            _R = ForwardDiff.jacobian(r, z)

            return _R
        end

        function check_variables(z)
            # system variables
            q3 = view(z, 1:3)
            n = z[4]
            b = view(z, 5:6)
            bs = view(z, 7:9)
            bz = view(z, 10:12)

            ϕs = z[13]

            if n <= 0.0
                return true
            end

            if ϕs <= 0.0
                return true
            end

            if !κ_so(bs)[2]
                println("bs not in cone")
                return true
            end

            if !κ_so(bz)[2]
                println("bz not in cone")
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

            # @show check_variables(z - α * Δ)
            # @error "STOP"

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
    z0_scale = 0.0, step_type = nothing)
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
model = Particle(1.0, 9.81, 0.5, 3)
h = 0.01

# initial conditions
v1 = [1.0; 10.0; 0.0]
q1 = [0.0; 0.0; 1.0]

v2 = v1 - gravity(model, q1) * h
q2 = q1 + 0.5 * (v1 + v2) * h

# v1 = [10.0; -20.0; 0.0]
q1 = [0.0; 0.0; 0.0]
q2 = [h * 5.0; h * 1.0; 0.0]

q_sol, y_sol, b_sol = simulate(q1, q2, 500, h)

# plot(hcat(q_sol...)[3:3, :]', xlabel = "", label = "z")
# plot!(h .* hcat(y_sol...)', xlabel = "", label = "n", linetype = :steppost)
#
# plot(hcat(q_sol...)[1:2, :]', xlabel = "", label = ["x" "y"])
# plot!(h * hcat(b_sol...)', label = ["b1" "b2"], linetype = :steppost)

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)


function visualize!(vis, model, q;
	Δt = 0.1, r = 0.1)

	default_background!(vis)
    setobject!(vis["particle"],
		Rect(Vec(0, 0, 0),Vec(2r, 2r, 2r)),
		MeshPhongMaterial(color = RGBA(1.0, 165.0 / 255.0, 0, 1.0)))

    anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

    for t = 1:length(q)
        MeshCat.atframe(anim, t) do
            settransform!(vis["particle"], Translation(q[t][1:3]...))
        end
    end

    MeshCat.setanimation!(vis, anim)
end

visualize!(vis, model,
    q_sol,
    Δt = h)

settransform!(vis["/Cameras/default"],
    compose(Translation(0.0, 0.0, -1.0), LinearMap(RotZ(pi))))

settransform!(vis["/Cameras/default"],
	compose(Translation(0.0, 0.0, 3.0),LinearMap(RotY(-pi/2.5))))
