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
        # println(pp)
        # println(i)
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
c8 = @SVector [r, r, r]
c2 = @SVector [r, r, -r]
c3 = @SVector [r, -r, r]
c4 = @SVector [r, -r, -r]
c5 = @SVector [-r, r, r]
c6 = @SVector [-r, r, -r]
c7 = @SVector [-r, -r, r]
c1 = @SVector [-r, -r, -r]


cr = @SVector [-r, -r, 0.0]
cl = @SVector [r, r, 0.0]

corner_offset = @SVector [cl, cr]
# corner_offset = @SVector [c1, c2, c3, c4, c5, c6, c7, c8]

model = Box(1.0, 1.0 / 12.0 * 1.0 * ((2.0 * r)^2 + (2.0 * r)^2),
 			1.0, 9.81,
            r, 2, corner_offset,
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

    # SVector{24}([(p + R * model.corner_offset[1])...,
    #           (p + R * model.corner_offset[2])...,
    #           (p + R * model.corner_offset[3])...,
    #           (p + R * model.corner_offset[4])...,
    #           (p + R * model.corner_offset[5])...,
    #           (p + R * model.corner_offset[6])...,
    #           (p + R * model.corner_offset[7])...,
    #           (p + R * model.corner_offset[8])...])
	SVector{6}([(p + R * model.corner_offset[1])...,
		(p + R * model.corner_offset[2])...])

end

function jacobian(model::Box, q)
	k(z) = kinematics(model, z)
	ForwardDiff.jacobian(k, q)
end

function ϕ_func(model::Box, q)
    p = view(q, 1:3)
    rot = view(q, 4:6)

    R = MRP(rot...)

    # @SVector [(p + R * model.corner_offset[1])[3],
    #           (p + R * model.corner_offset[2])[3],
    #           (p + R * model.corner_offset[3])[3],
    #           (p + R * model.corner_offset[4])[3],
    #           (p + R * model.corner_offset[5])[3],
    #           (p + R * model.corner_offset[6])[3],
    #           (p + R * model.corner_offset[7])[3],
    #           (p + R * model.corner_offset[8])[3]]
	SVector{2}([(p + R * model.corner_offset[1])[3],
		(p + R * model.corner_offset[2])[3]])
end

# dynamics
function dynamics(model, q1, q2, q3, λ, h)
      nq = model.nq
      SVector{nq}(mass_matrix(model) * (2.0 * q2 - q1 - q3) / h
            - h * gravity(model)
            + h * jacobian(model, q3)' * λ)
end

mrp = MRP(UnitQuaternion(RotY(0.0) * RotX(0.0)))

q1 = [0.0; 0.0; 0.5; mrp.x; mrp.y; mrp.z]
q2 = [0.0; 0.0; 0.5; mrp.x; mrp.y; mrp.z]
h = 0.1

ϕ_func(model, q2)

function r_action(z)
    q3 = z[1:6]
    ϕ = ϕ_func(model, q3)
	# λ = z[7:8]
	# f = [0.0; 0.0; λ[1]; 0.0; 0.0; λ[2]]
    λ = z[7:8]
	s = z[9:10]
	f = zeros(eltype(z), 6)

	for i = 1:model.n_corners
		f[i * 3] = λ[i]
	end

    [dynamics(model, q1, q2, q3, f, h);
	 s - ϕ
     s - κ_no(s - λ)]
end

function solve()
    z = rand(10)
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
function solve(q1, q2, h; init = 0.0, step = :ls)

    z = init * rand(24)
    z[1:6] = copy(q2)

	function r_if(z)
		q3 = z[1:6]
		ϕ = ϕ_func(model, q3)
		λ = z[7:8]
		s = z[9:10]
		b̄ = z[11:16]
		μ = z[17:18]
		η = z[19:24]

		# velocities
		v = jacobian(model, q3) * (q3 - q2) ./ h

		# contact forces
		f = zeros(eltype(z), 6)

		for i = 1:model.n_corners
			f[i * 3] = λ[i]
			f[(i-1) * 3 .+ (1:2)] = b̄[(i-1) * 3 .+ (1:2)]
		end

		[dynamics(model, q1, q2, q3, f, h); # 1:6
		 s - ϕ;
		 s - κ_no(s - λ);
		 [v[1:2]; -μ[1]] - η[1:3];
		 λ[1] - b̄[3];
		 b̄[1:3] - κ_soc(b̄[1:3] - η[1:3]);
		 [v[4:5]; -μ[2]] - η[4:6];
	 	 λ[2] - b̄[6];
	 	 b̄[4:6] - κ_soc(b̄[4:6] - η[4:6])]
	end

	function R_if(z)
		_R = ForwardDiff.jacobian(r_if, z)

		# fix projection
		I2 = Diagonal(ones(2))
		I3 = Diagonal(ones(3))

		λ = z[7:8]
		s = z[9:10]
		b̄ = z[11:16]
		μ = z[17:18]
		η = z[19:24]

		Jno = Jκ_no(s - λ)
		Jsoc1 = Jκ_soc(b̄[1:3] - η[1:3])
		Jsoc2 = Jκ_soc(b̄[4:6] - η[4:6])

		_R[9:10, 9:10] = I2 - Jno
		_R[9:10, 7:8] = Jno
		_R[15:17, 11:13] = I3 - Jsoc1
		_R[15:17, 19:21] = Jsoc1
		_R[22:24, 14:16] = I3 - Jsoc2
		_R[22:24, 22:24] = Jsoc2

		return _R
	end

    extra_iters = 0

    for i = 1:500
        _F = r_if(z)
		if norm(_F) < 1.0e-8
			println("iter ($i) - norm: $(norm(r_if(z)))")
	        return z, true
		end

        _J = R_if(z)
		if step == :ls
        	Δ = (_J' * _J + 1.0e-6 * I) \ (_J' * _F)
		else
			Δ = gmres(_J, 1.0 * _F, abstol = 1.0e-12, maxiter = i + extra_iters)
		end
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

    if norm(r_if(z)) < 1.0e-8
        status = true
    else
        status = false
    end

    return z, status
end

h = 0.1
mrp = MRP(UnitQuaternion(RotY(pi / 3.0) * RotX(pi / 6.0)))

v1 = [1.0; -1.0; 0.0; 0.0; 0.0; 0.0]
q2 = [0.0; 0.0; 1.0; mrp.x; mrp.y; mrp.z]
q1 = q2 - h * v1

ϕ_func(model, q1)
z_sol, = solve(q1, q2, h, init = 0.0)
# z_sol[23:25]

function simulate(q1, q2, T, h)
    println("simulation")
    q = [q1, q2]
    y = [zeros(8)]
    b = [zeros(16)]
    for t = 1:T
        println("   t = $t")
        z_sol, status = solve(q[end-1], q[end], h, init = 0.001, step = :ls)

        if !status
            @warn "failed step (t = $t)"
			z_sol, status = solve(q[end-1], q[end], h, init = 0.001, step = :gmres)

			if !status
				@error "failed step (t = $t) [again]"
				println(ϕ_func(model, q[end]))
				println(ϕ_func(model, z_sol[1:6]))
            	return q, y, b
			end
        else
            nothing
        end
		push!(q, z_sol[1:6])
		# push!(y, z_sol[4])
		# push!(b, z_sol[5:6])
    end

    return q, y, b
end

q_sol, y_sol, b_sol = simulate(q1, q2, 100, h)
q_sol[end]
q_sol[end-1]
plot(hcat(q_sol...)[3:3, :]', xlabel = "")
plot(h .* hcat(y_sol...)', xlabel = "", linetype = :steppost)
plot(hcat([ϕ_func(model, q) for q in q_sol]...)', xlabel = "")
#
plot(hcat(q_sol...)[1:2, :]', xlabel = "")
plot(h * hcat(b_sol...)', linetype = :steppost)
