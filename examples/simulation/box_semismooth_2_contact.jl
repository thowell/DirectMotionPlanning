using LinearAlgebra, ForwardDiff, Distributions, StaticArrays
using IterativeSolvers
using Plots
using Rotations

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

# impact problem
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

# Kinematics
r = 0.5
c1 = @SVector [r, r, r]
c2 = @SVector [r, r, -r]
c3 = @SVector [r, -r, r]
c4 = @SVector [r, -r, -r]
c5 = @SVector [-r, r, r]
c6 = @SVector [-r, r, -r]
c7 = @SVector [-r, -r, r]
c8 = @SVector [-r, -r, -r]

corner_offset = @SVector [c6]#, c8]
# corner_offset = @SVector [c1, c2, c3, c4, c5, c6, c7, c8]

model = Box(1.0, 1.0 / 12.0 * 1.0 * ((2.0 * r)^2 + (2.0 * r)^2),
 			1.0, 9.81,
            r, 1, corner_offset,
            6)

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
    rot = view(q, 4:6)

    R = MRP(rot...)

    SVector{3}([(p + R * model.corner_offset[1])...])#,
              # (p + R * model.corner_offset[2])...])
              # (p + R * model.corner_offset[3])...,
              # (p + R * model.corner_offset[4])...,
              # (p + R * model.corner_offset[5])...,
              # (p + R * model.corner_offset[6])...,
              # (p + R * model.corner_offset[7])...,
              # (p + R * model.corner_offset[8])...])
	# SVector{6}([(p + R * model.corner_offset[1])...,
	# 	(p + R * model.corner_offset[2])...])

end


function jacobian(model::Box, q)
	k(z) = kinematics(model, z)
	ForwardDiff.jacobian(k, q)
end

function ϕ_func(model::Box, q)
    p = view(q, 1:3)
    rot = view(q, 4:6)

    R = MRP(rot...)

    @SVector [(p + R * model.corner_offset[1])[3]]#
              # (p + R * model.corner_offset[2])[3]]
              # (p + R * model.corner_offset[3])[3],
              # (p + R * model.corner_offset[4])[3],
              # (p + R * model.corner_offset[5])[3],
              # (p + R * model.corner_offset[6])[3],
              # (p + R * model.corner_offset[7])[3],
              # (p + R * model.corner_offset[8])[3]]
	# SVector{2}([(p + R * model.corner_offset[1])[3],
	# 	(p + R * model.corner_offset[2])[3]])
end

# dynamics
function dynamics(model, q1, q2, q3, λ, h)
      nq = model.nq
      SVector{nq}(mass_matrix(model) * (2.0 * q2 - q1 - q3) / h
            - h * gravity(model)
            + h * jacobian(model, q3)' * λ)
end

mrp = MRP(UnitQuaternion(RotY(0.0) * RotX(0.0)))

q1 = [0.0; 0.0; 0.6; mrp.x; mrp.y; mrp.z]
q2 = [0.0; 0.0; 0.55; mrp.x; mrp.y; mrp.z]
h = 0.1

ϕ_func(model, q2)

function r_action(z)
    q3 = z[1:6]
    ϕ = ϕ_func(model, q3)
	# λ = z[7:8]
	# f = [0.0; 0.0; λ[1]; 0.0; 0.0; λ[2]]
    λ = z[7:7]
	f = zeros(eltype(z), 3)

	for i = 1:model.n_corners
		f[i * 3] = λ[i]
	end

    [dynamics(model, q1, q2, q3, f, h);
     ϕ - κ_no(ϕ - λ)]
end

function solve()
    z = zeros(7)
	z[1:6] = q2

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
ϕ_func(model, z_sol[1:6])
# impact and friction


function solve(q1, q2, h)

    z = 0.0 * rand(14)
    z[1:6] = copy(q2)

	function r_if(z)
		q3 = z[1:6]
		ϕ = ϕ_func(model, q3)
		λ = z[7:7]
		b̄ = z[8:10]
		μ = z[11:11]
		η = z[12:14]

		# velocities
		v = jacobian(model, q3) * (q3 - q2) ./ h

		# contact forces
		f = [b̄[1:2]; λ[1]]#; b̄[4:5]; λ[2]]

		[dynamics(model, q1, q2, q3, f, h); # 1:6
		 ϕ - κ_no(ϕ - λ); # 7:7

		 [v[1:2]; -μ[1]] - η[1:3];         # 8:10
		  λ[1] - b̄[3];                     # 11
		  b̄[1:3] - κ_soc(b̄[1:3] - η[1:3]); # 12:14

		  # [v[4:5]; -μ[2]] - η[4:6];        # 16:18
	 	  # λ[2] - b̄[6];                     # 19
	 	  # b̄[4:6] - κ_soc(b̄[4:6] - η[4:6]); # 20:22
		  #
		  # [v[7:8]; -μ[3]] - η[7:9];        # 29:31
		  # λ[3] - b̄[9]; 					# 32
		  # b̄[7:9] - κ_soc(b̄[7:9] - η[7:9]); # 33:35
		  #
		  # [v[10:11]; -μ[4]] - η[10:12];          # 36:38
		  # λ[4] - b̄[12];                          # 39
		  # b̄[10:12] - κ_soc(b̄[10:12] - η[10:12]); # 40:42
		  #
		  # [v[13:14]; -μ[5]] - η[13:15];          # 43:45
		  # λ[5] - b̄[15];                          # 46
		  # b̄[13:15] - κ_soc(b̄[13:15] - η[13:15]); # 47:49
		  #
		  # [v[16:17]; -μ[6]] - η[16:18];          # 50:52
		  # λ[6] - b̄[18];                          # 53
		  # b̄[16:18] - κ_soc(b̄[16:18] - η[16:18]); # 54:56
		  #
		  # [v[19:20]; -μ[7]] - η[19:21];          # 57:59
		  # λ[7] - b̄[21];                          # 60
		  # b̄[19:21] - κ_soc(b̄[19:21] - η[19:21]); # 61:63
		  #
		  # [v[22:23]; -μ[8]] - η[22:24];          # 64:66
		  # λ[8] - b̄[24];                          # 67
		  # b̄[22:24] - κ_soc(b̄[22:24] - η[22:24]); # 68:70
		  ]
	end

	function R_if(z)
		_R = ForwardDiff.jacobian(r_if, z)

		# fix projection
		I3 = Diagonal(ones(3))

		b̄_idx = (8:10)
		η_idx = (12:14)
		b̄ = z[b̄_idx]
		η = z[η_idx]

		J1 = Jκ_soc(b̄[1:3] - η[1:3])
		# J2 = Jκ_soc(b̄[3 .+ (1:3)] - η[3 .+ (1:3)])
		# J3 = Jκ_soc(b̄[6 .+ (1:3)] - η[6 .+ (1:3)])
		# J4 = Jκ_soc(b̄[9 .+ (1:3)] - η[9 .+ (1:3)])
		# J5 = Jκ_soc(b̄[12 .+ (1:3)] - η[12 .+ (1:3)])
		# J6 = Jκ_soc(b̄[15 .+ (1:3)] - η[15 .+ (1:3)])
		# J7 = Jκ_soc(b̄[18 .+ (1:3)] - η[18 .+ (1:3)])
		# J8 = Jκ_soc(b̄[21 .+ (1:3)] - η[21 .+ (1:3)])

		_R[12:14, b̄_idx[1:3]] = I3 - J1
		_R[12:14, η_idx[1:3]] = J1

		# _R[20:22, b̄_idx[3 .+ (1:3)]] = I3 - J2
		# _R[20:22, η_idx[3 .+ (1:3)]] = J2

		# _R[33:35, b̄_idx[6 .+ (1:3)]] = I3 - J3
		# _R[33:35, η_idx[6 .+ (1:3)]] = J3
		#
		#
		# _R[40:42, b̄_idx[9 .+ (1:3)]] = I3 - J4
		# _R[40:42, η_idx[9 .+ (1:3)]] = J4
		#
		# _R[47:49, b̄_idx[12 .+ (1:3)]] = I3 - J5
		# _R[47:49, η_idx[12 .+ (1:3)]] = J5
		#
		# _R[54:56, b̄_idx[15 .+ (1:3)]] = I3 - J6
		# _R[54:56, η_idx[15 .+ (1:3)]] = J6
		#
		# _R[61:63, b̄_idx[18 .+ (1:3)]] = I3 - J7
		# _R[61:63, η_idx[18 .+ (1:3)]] = J7
		#
		# _R[68:70, b̄_idx[21 .+ (1:3)]] = I3 - J8
		# _R[68:70, η_idx[21 .+ (1:3)]] = J8

		return _R
	end

    extra_iters = 0

    for i = 1:100
        _F = r_if(z)
        _J = R_if(z)
        Δ = gmres(_J, 1.0 * _F, abstol = 1.0e-12, maxiter = i + extra_iters)
        # Δ = (_J' * _J + 1.0e-6 * I) \ (_J' * _F)
        iter = 0
        α = 1.0
        while norm(r_if(z - α * Δ))^2.0 >= (1.0 - 0.001 * α) * norm(_F)^2.0 && α > 1.0e-4
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

        println("iter ($i) - norm: $(norm(r_if(z)))")

        z .-= α * Δ
    end

    if norm(r_if(z)) < 1.0e-5
        status = true
    else
        status = false
    end

    return z, status
end

h = 0.1
mrp = MRP(UnitQuaternion(RotY(0.0) * RotX(0.0)))

v1 = [0.0; 0.0; 0.0; 0.0; 0.0; 0.0]
q2 = [0.0; 0.0; 0.525; mrp.x; mrp.y; mrp.z]
q1 = q2 - h * v1

ϕ_func(model, q2)
z_sol,  = solve(q1, q2, h)
ϕ_func(model, z_sol[1:6])



function simulate(q1, q2, T, h)
    println("simulation")
    q = [q1, q2]
    y = [zeros(8)]
    b = [zeros(16)]
    for t = 1:T
        println("   t = $t")
        z_sol, status = solve(q[end-1], q[end], h)

        if !status
            @error "failed step (t = $t)"
            return q, y, b
        else
            push!(q, z_sol[1:6])
            # push!(y, z_sol[4])
            # push!(b, z_sol[5:6])
        end
    end

    return q, y, b
end

q_sol, y_sol, b_sol = simulate(q1, q2, 10, h)
# q_sol[end]
# q_sol[end-1]
plot(hcat(q_sol...)[3:3, :]', xlabel = "")
plot(h .* hcat(y_sol...)', xlabel = "", linetype = :steppost)

plot(hcat(q_sol...)[1:2, :]', xlabel = "")
plot(h * hcat(b_sol...)', linetype = :steppost)
