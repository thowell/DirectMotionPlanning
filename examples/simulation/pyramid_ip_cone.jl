using LinearAlgebra, ForwardDiff, StaticArrays
using Plots, Rotations

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

function cone_product(z, s)
    [z' * s; z[1] * s[2:3] + s[1] * z[2:3]]
end

"""
    box dynamics
    - 3D particle with orientation and 8 corners subject to contact forces

    - configuration: q = (px, py, pz, rx, ry, rz) ∈ R⁶
        - orientation : modified Rodrigues angles

    - contacts (8x)
        - impact force (magnitude): n ∈ R₊
        - friction force: b ∈ R²
            - contact force: λ = (b, n) ∈ R² × R₊
            - friction coefficient: μ ∈ R₊

    Discrete Mechanics and Variational Integrators
        pg. 363
"""

struct Box
    m # mass
    J # inertia
    μ # friction coefficient
    g # gravity

    r             # corner length
	n_corners     # number of corners
    corner_offset # precomputed corner offsets

    nq # configurations dimension
end

# Methods
function mass_matrix(model::Box)
	Diagonal(@SVector [model.m, model.m, model.m,
		model.J, model.J, model.J])
end

function gravity(model::Box)
	@SVector [0., 0., model.m * model.g, 0., 0., 0.]
end

function kinematics(model::Box, q)
    p = view(q, 1:3)
    r = view(q, 4:6)

    R = MRP(r...)

    SVector{24}([(p + R * model.corner_offset[1])...,
                 (p + R * model.corner_offset[2])...,
                 (p + R * model.corner_offset[3])...,
                 (p + R * model.corner_offset[4])...,
                 (p + R * model.corner_offset[5])...,
                 (p + R * model.corner_offset[6])...,
                 (p + R * model.corner_offset[7])...,
                 (p + R * model.corner_offset[8])...])
end

function jacobian(model::Box, q)
	k(z) = kinematics(model, z)
	ForwardDiff.jacobian(k, q)
end

function signed_distance(model::Box, q)
	idx = collect([3, 6, 9, 12, 15, 18, 21, 24])
	kinematics(model, q)[idx]
end

function P_func(model::Box, q)
	idx = collect([1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23])
	k(z) = kinematics(model, z)[idx]
	ForwardDiff.jacobian(k, q)
end

# dynamics
function dynamics(model, q1, q2, q3, λ, h)
      nq = model.nq
      SVector{nq}(mass_matrix(model) * (2.0 * q2 - q1 - q3) / h
            - h * gravity(model)
            + h * jacobian(model, q3)' * λ)
end

# Kinematics
num_contacts = 8
r = 0.5
c1 = @SVector [0.0, 0.0, 2.0 * r]
c2 = @SVector [r, r, 0.0]
c3 = @SVector [r, -r, r]
c4 = @SVector [r, -r, 0.0]
c5 = @SVector [-r, r, r]
c6 = @SVector [-r, r, 0.0]
c7 = @SVector [-r, -r, r]
c8 = @SVector [-r, -r, 0.0]

corner_offset = @SVector [c2, c4, c6, c8, c1, c3, c5, c7]

# Model
nq = 6
model = Box(1.0,
			1.0 / 12.0 * 1.0 * ((2.0 * r)^2 + (2.0 * r)^2),
 			0.5,
			9.81,
            r, num_contacts, corner_offset,
            nq)

# qq = rand(nq)
# kinematics(model, qq)
# ϕ_func(model, qq)
# jacobian(model, qq)
# P_func(model, qq)


# var
num_var = 6 + num_contacts + num_contacts + 2 * num_contacts + 3 * num_contacts + 3 * num_contacts
function unpack(z)
	q = view(z, 1:6)
	n = view(z, 6 .+ (1:num_contacts))
	ϕs = view(z, 6 + num_contacts .+ (1:num_contacts))
	b = view(z, 6 + num_contacts + num_contacts .+ (1:2 * num_contacts))
	bs = view(z, 6 + num_contacts + num_contacts + 2 * num_contacts .+ (1:3 * num_contacts))
	bz = view(z, 6 + num_contacts + num_contacts + 2 * num_contacts + 3 * num_contacts .+ (1:3 * num_contacts))

	b_traj = [b[(i - 1) * 2 .+ (1:2)] for i = 1:num_contacts]
	bs_traj = [bs[(i - 1) * 3 .+ (1:3)] for i = 1:num_contacts]
	bz_traj = [bz[(i - 1) * 3 .+ (1:3)] for i = 1:num_contacts]

	return q, n, ϕs, b_traj, bs_traj, bz_traj
end

function initialize(q2, num_var)
	z = 0.01 * ones(num_var)

	z[1:6] = copy(q2)
	z[6 + num_contacts + num_contacts + 2 * num_contacts .+ (1:3 * num_contacts)] = vcat([[1.0; 0.01; 0.01] for i = 1:num_contacts]...)
	z[6 + num_contacts + num_contacts + 2 * num_contacts + 3 * num_contacts .+ (1:3 * num_contacts)] = vcat([[1.0; 0.01; 0.01] for i = 1:num_contacts]...)

	return z
end
# z0 = initialize(rand(6), num_var)
# num_var
# z0 = rand(num_var)
# unpack(z0)

"""
    step particle system
        solves 1-step feasibility problem
"""
function _step(q1, q2, h;
    tol = 1.0e-8, z0_scale = 0.001, step_type = :none, max_iter = 100)
    # 1-step optimization problem:
    #     find z
    #     s.t. r(z) = 0
    #
    # z = (q, n, ϕs, b, bs, bz)
    #     s are slack variables for convenience

    # initialize
    z = initialize(q2, num_var)

    ρ = 1.0 # barrier parameter
    flag = false

    for k = 1:6
        function r(z)
            # system variables
			# q3, n, ϕs, b = unpack(z)
			q3, n, ϕs, b, bs, bz = unpack(z)


            λ = [b[1];
				 n[1];
			     b[2];
				 n[2];
				 b[3];
				 n[3];
				 b[4];
				 n[4];
				 b[5];
				 n[5];
				 b[6];
				 n[6];
				 b[7];
				 n[7];
				 b[8];
				 n[8]]

			# contact forces
            ϕ = signed_distance(model, q3) # signed-distance function
            vT = P_func(model, q3) * (q3 - q2) ./ h

            # G = [zeros(1, 2); -Diagonal(ones(2))]
            # g = [model.μ * n; zeros(2)]

            # W = W̄(bz, bs)
            # Winv = W̄inv(bz, bs)

            # action optimality conditions
            [dynamics(model, q1, q2, q3, λ, h);
             ϕs - ϕ;
             n .* ϕs .- ρ;

             # maximum dissipation optimality conditions
			 vT[1:2] - bz[1][2:3];
			 vT[3:4] - bz[2][2:3];
			 vT[5:6] - bz[3][2:3];
			 vT[7:8] - bz[4][2:3];
			 vT[9:10] - bz[5][2:3];

             # vT - [bz[1][2:3];
			 #       bz[2][2:3];
				#    bz[3][2:3];
				#    bz[4][2:3];
				#    bz[5][2:3];
				#    bz[6][2:3];
				#    bz[7][2:3];
				#    bz[8][2:3]];
			  # b[1];
			  # b[2];
			  # b[3];
			  # b[4];
			  # b[5];
			  b[6];
			  b[7];
			  b[8];
			  # bs[1];
			  # bs[2];
			  # bs[3];
			  # bs[4];
			  # bs[5];
			  bs[6];
			  bs[7];
			  bs[8];
			  # bz[1];
			  # bz[2];
			  # bz[3];
			  # bz[4];
			  # bz[5];
			  bz[6];
			  bz[7];
			  bz[8];
			  bs[1] - [model.μ * n[1]; b[1]];
			  bs[2] - [model.μ * n[2]; b[2]];
			  bs[3] - [model.μ * n[3]; b[3]];
			  bs[4] - [model.μ * n[4]; b[4]];
			  bs[5] - [model.μ * n[5]; b[5]];

             # [bs[1];
			 #  bs[2];
			 #  bs[3];
			 #  bs[4];
			 #  bs[5];
			 #  bs[6];
			 #  bs[7];
			 #  bs[8]]
			 #  - [model.μ * n[1]; b[1];
			 #       model.μ * n[2]; b[2];
				#    model.μ * n[3]; b[3];
				#    model.μ * n[4]; b[4];
				#    model.μ * n[5]; b[5];
				#    model.μ * n[6]; b[6];
				#    model.μ * n[7]; b[7];
				#    model.μ * n[8]; b[8]];
             cone_product(bz[1], bs[1]) - ρ * [1.0; 0.0; 0.0];
			 cone_product(bz[2], bs[2]) - ρ * [1.0; 0.0; 0.0];
			 cone_product(bz[3], bs[3]) - ρ * [1.0; 0.0; 0.0];
			 cone_product(bz[4], bs[4]) - ρ * [1.0; 0.0; 0.0]
			 cone_product(bz[5], bs[5]) - ρ * [1.0; 0.0; 0.0]]
			 # cone_product(bz[3], bs[3]) - ρ * [1.0; 0.0; 0.0];
			 # cone_product(bz[4], bs[4]) - ρ * [1.0; 0.0; 0.0];
			 # cone_product(bz[5], bs[5]) - ρ * [1.0; 0.0; 0.0];
			 # cone_product(bz[6], bs[6]) - ρ * [1.0; 0.0; 0.0];
			 # cone_product(bz[7], bs[7]) - ρ * [1.0; 0.0; 0.0];
			 # cone_product(bz[8], bs[8]) - ρ * [1.0; 0.0; 0.0]]
        end

        # Jacobian
        function R(z)
            # differentiate r
            _R = ForwardDiff.jacobian(r, z)

            return _R
        end

        function check_variables(z)
            # system variables
            _, n, ϕs, _, bs, bz = unpack(z)
			# _, n, ϕs, _ = unpack(z)

            if any([ni <= 0.0 for ni in n])
                return true
            end

            if any([ϕsi <= 0.0 for ϕsi in ϕs])
                return true
            end

            if any([!κ_so(bsi)[2] for bsi in bs])
				# println("bs not in cone")

				# for bsi in bs
				# 	println(bsi)
				# end
                return true
            end

            if any([!κ_so(bzi)[2] for bzi in bz])
                # println("bz not in cone")
                return true
            end

            return false
        end

        extra_iters = 0

		# @show check_variables(z)

        for i = 1:max_iter
            # compute residual, residual Jacobian
            res = r(z)
			# println("res: $(norm(res, Inf))")
            if norm(res) < tol
                # println("   iter ($i) - norm: $(norm(res))")
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
				# println("ITER: $iter")
				# println("+res: $(norm(r(z - α * Δ), Inf))")

                α = 0.5 * α
                # println("   α = $α")
                iter += 1
                if iter > 50
                    @error "backtracking line search fail"
                    flag = false
                    return z, false
                end
            end

			# println("ESCAPE!!!")
			# for bsi in unpack(z - α * Δ)[5]
			# 	println(bsi)
			# end

			while norm(r(z - α * Δ), Inf) >= norm(res, Inf)
				# println("+res: $(norm(r(z - α * Δ), Inf))")
            # while norm(r(z - α * Δ))^2.0 >= (1.0 - 0.001 * α) * norm(res)^2.0
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
        # println("ρ: $ρ")
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
    n = [zeros(8)]
    b = [zeros(2 * 8)]

    # step
    for t = 1:T
        println("   t = $t")
        z_sol, status = _step(q[end-1], q[end], h,
            z0_scale = z0_scale, step_type = step_type)
        if !status
            @error "failed step (t = $t)"
            return q, n, b
        else
			q3, nt, _, bt, _, _ = unpack(z_sol)
            push!(q, q3)
            push!(n, nt)
            push!(b, vcat(bt...))
        end
    end

    return q, n, b
end

# simulation setup
# model
h = 0.05

# initial conditions
mrp = MRP(UnitQuaternion(RotY(pi / 6.0) * RotX(pi / 4.0)))

v1 = [-7.5; -1.0; 0.0; 0.0; 0.0; 0.0]
q1 = [0.0; 0.0; 1.0; mrp.x; mrp.y; mrp.z]

v2 = v1 - gravity(model) * h
q2 = q1 + 0.5 * (v1 + v2) * h

signed_distance(model, q2)
q_sol, y_sol, b_sol = simulate(q1, q2, 100, h)

y_sol[end]
signed_distance(model, q_sol[end])
kinematics(model, q_sol[end])
# unpack(z_sol)[5][1]
# plot(hcat(q_sol...)[3:3, :]', xlabel = "", label = "z")
# plot!(h .* hcat(y_sol...)', xlabel = "", label = "n", linetype = :steppost)
#
# plot(hcat(q_sol...)[1:2, :]', xlabel = "", label = ["x" "y"])
# plot!(h * hcat(b_sol...)', label = ["b1" "b2"], linetype = :steppost)

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)

#
function visualize!(vis, model::Box, q;
        Δt = 0.1)

	default_background!(vis)

    # setobject!(vis["box"], GeometryBasics.Rect(Vec(-1.0 * model.r,
	# 	-1.0 * model.r,
	# 	-1.0 * model.r),
	# 	Vec(2.0 * model.r, 2.0 * model.r, 2.0 * model.r)),
	# 	MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

	pyramid = Pyramid(Point3(0.0, 0.0, 0.0), 2.0 * model.r, 2.0 * model.r)
	setobject!(vis["pyramid"], pyramid,
		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

    for i = 1:5
        setobject!(vis["corner$i"], GeometryBasics.Sphere(Point3f0(0),
            convert(Float32, 0.05)),
            MeshPhongMaterial(color = RGBA(1.0, 165.0 / 255.0, 0.0, 1.0)))
    end

    anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

    for t = 1:length(q)
        MeshCat.atframe(anim, t) do

            settransform!(vis["pyramid"],
				compose(Translation(q[t][1:3]...), LinearMap(MRP(q[t][4:6]...))))

            for i = 1:5
                settransform!(vis["corner$i"],
                    Translation((q[t][1:3] + MRP(q[t][4:6]...) * (corner_offset[i]))...))
            end
        end
    end
    # settransform!(vis["/Cameras/default"], compose(Translation(-1, -1, 0),LinearMap(RotZ(pi/2))))
    MeshCat.setanimation!(vis, anim)
end

visualize!(vis, model,
    q_sol,
    Δt = h)
open(vis)
settransform!(vis["/Cameras/default"],
    compose(Translation(0.0, 0.0, -1.0), LinearMap(RotZ(pi))))

settransform!(vis["/Cameras/default"],
	compose(Translation(0.0, 0.0, 3.0),LinearMap(RotY(-pi/2.5))))
