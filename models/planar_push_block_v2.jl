"""
    planar push block
        particle with contacts at each corner
"""
struct PlanarPushBlock{I, T} <: Model{I, T}
    n::Int
    m::Int
    d::Int

    mass
    J
    μ
    g

    contact_corner_offset
	control_input_offset

    nq
    nu
    nc
    nf
    nb

    idx_u
    idx_λ
    idx_b
    idx_ψ
    idx_η
    idx_s
end

# Dimensions
nq = 3 # configuration dimension
nu = 2 * 4 + 4 # control dimension
nc = 4 # number of contact points
nf = 4 # number of faces for friction cone pyramid
nb = nc * nf
ns = 1

n = 2 * nq
m = nu + nc + nb + nc + nb + 1
d = 0

idx_u = (1:nu)
idx_λ = nu .+ (1:nc)
idx_b = nu + nc .+ (1:nb)
idx_ψ = nu + nc + nb .+ (1:nc)
idx_η = nu + nc + nb + nc .+ (1:nb)
idx_s = nu + nc + nb + nc + nb .+ (1:ns)

# Kinematics
r = 0.1

# contact corner
cc1 = @SVector [r, r]
cc2 = @SVector [-r, r]
cc3 = @SVector [r, -r]
cc4 = @SVector [-r, -r]

# control location
cu5 = @SVector [r, 0.0]
cu6 = @SVector [0.0, r]
cu7 = @SVector [-r, 0.0]
cu8 = @SVector [0.0, -r]

contact_corner_offset = @SVector [cc1, cc2, cc3, cc4]
control_input_offset = @SVector [cu5, cu6, cu7, cu8]

# Parameters
μ = 1.0  # coefficient of friction
g = 9.81
mass = 1.0   # mass
J = 1.0 / 12.0 * mass * ((2.0 * r)^2 + (2.0 * r)^2)

# Methods
M_func(model::PlanarPushBlock, q) = Diagonal(@SVector [model.mass, model.mass, model.J])

function C_func(model::PlanarPushBlock, q, q̇)
	SVector{3}([0.0, 0.0, 0.0])
end

function rotation_matrix(x)
	SMatrix{2,2}([cos(x) -sin(x); sin(x) cos(x)])
end

function ϕ_func(model::PlanarPushBlock, q)
    p = view(q, 1:2)
	θ = q[3]
    R = rotation_matrix(θ)

    @SVector [0.0, 0.0, 0.0, 0.0]
end

function control_kinematics_func(model::PlanarPushBlock, q)
    p = q[1:2]
	θ = q[3]
    R = rotation_matrix(θ)

    SVector{8}([p + R * model.control_input_offset[1];
              p + R * model.control_input_offset[2];
              p + R * model.control_input_offset[3];
              p + R * model.control_input_offset[4]])
              # p + R * model.control_input_offset[5];
              # p + R * model.control_input_offset[6];
              # p + R * model.control_input_offset[7];
              # p + R * model.control_input_offset[8]])
end

function B_func(model::PlanarPushBlock, q, u)
	tmp(z) = control_kinematics_func(model, z)
	ForwardDiff.jacobian(tmp, q)

	mask = Diagonal([1, 0, 1, 0, 1, 0, 1, 0])

	R1 = rotation_matrix(-1.0 * π)
	R2 = rotation_matrix(-0.5 * π)
	R3 = rotation_matrix(0.0)
	R4 = rotation_matrix(0.5 * π)
	R = cat(R1, R2, R3, R4, dims=(1, 2))

	T1 = rotation_matrix(u[2])
	T2 = rotation_matrix(u[4])
	T3 = rotation_matrix(u[6])
	T4 = rotation_matrix(u[8])
	T = cat(T1, T2, T3, T4, dims=(1, 2))

	transpose(ForwardDiff.jacobian(tmp, q)) * T * R * mask * u
end

function N_func(model::PlanarPushBlock, q)
    tmp(z) = ϕ_func(model, z)
    ForwardDiff.jacobian(tmp, q)
end

function P_func(model::PlanarPushBlock, q)
    map = [1. 0.;
           0. 1.;
           -1. 0.;
           0. -1.]

    function p(x)
        pos = view(x, 1:2)
		θ = x[3]
        R = rotation_matrix(θ)

        [map * (pos + R * model.contact_corner_offset[1])[1:2];
         map * (pos + R * model.contact_corner_offset[2])[1:2];
         map * (pos + R * model.contact_corner_offset[3])[1:2];
         map * (pos + R * model.contact_corner_offset[4])[1:2]]
    end

    ForwardDiff.jacobian(p, q)
end

function friction_cone(model::PlanarPushBlock,u)
    λ = u[model.idx_λ]
    b = u[model.idx_b]

    @SVector [model.μ[1] * λ[1] - sum(b[1:4]),
              model.μ[2] * λ[2] - sum(b[5:8]),
              model.μ[3] * λ[3] - sum(b[9:12]),
              model.μ[4] * λ[4] - sum(b[13:16])]
end

function maximum_dissipation(model::PlanarPushBlock, x⁺, u, h)
    q3 = x⁺[model.nq .+ (1:model.nq)]
	q2 = x⁺[1:model.nq]

    ψ = u[model.idx_ψ]
    ψ_stack = [ψ[1] * ones(4);
               ψ[2] * ones(4);
               ψ[3] * ones(4);
               ψ[4] * ones(4)]

    η = u[model.idx_η]

    P_func(model, q3) * (q3 - q2) / h + ψ_stack - η
end

function lagrangian_derivatives(model, q, v)
	D1L = -1.0 * C_func(model, q, v)
    D2L = M_func(model, q) * v
	return D1L, D2L
end


function fd(model::PlanarPushBlock{Discrete, FixedTime}, x⁺, x, u, w, h, t)
	q3 = view(x⁺, model.nq .+ (1:model.nq))
	q2⁺ = view(x⁺, 1:model.nq)
	q2⁻ = view(x, model.nq .+ (1:model.nq))
	q1 = view(x, 1:model.nq)
	u_ctrl = view(u, model.idx_u)
	λ = view(u, model.idx_λ)
	b = view(u, model.idx_b)

	qm1 = 0.5 * (q1 + q2⁺)
    vm1 = (q2⁺ - q1) / h
    qm2 = 0.5 * (q2⁺ + q3)
    vm2 = (q3 - q2⁺) / h

	D1L1, D2L1 = lagrangian_derivatives(model, qm1, vm1)
	D1L2, D2L2 = lagrangian_derivatives(model, qm2, vm2)

	[q2⁺ - q2⁻;
     (0.5 * h * D1L1 + D2L1 + 0.5 * h * D1L2 - D2L2
     + B_func(model, qm2, u_ctrl[1:8])
     + transpose(P_func(model, q3)) * SVector{16}(b))]
end

model = PlanarPushBlock{Discrete, FixedTime}(n, m, d,
			mass, J, [[μ for i = 1:nc]..., 0.1], g,
			contact_corner_offset,
			control_input_offset,
            nq, nu, nc, nf, nb,
            idx_u,
            idx_λ,
            idx_b,
            idx_ψ,
            idx_η,
            idx_s)


# function control_input(q, u)
# 	idx = [(i - 1) * 2 .+ (1:2) for i = 1:4]
#
#     k = control_kinematics_func(model, q)
# 	k_input = u[model.idx_u][9:10]
#
# 	d1 = norm(k[1:2] - k_input)
# 	d2 = norm(k[3:4] - k_input)
# 	d3 = norm(k[5:6] - k_input)
# 	d4 = norm(k[7:8] - k_input)
# 	# d5 = norm(k[9:10] - k_input)
# 	# d6 = norm(k[11:12] - k_input)
# 	# d7 = norm(k[13:14] - k_input)
# 	# d8 = norm(k[15:16] - k_input)
#
# 	_, min_idx = findmin([d1, d2, d3, d4])#, d5, d6, d7, d8])
#
# 	return u[idx[min_idx]], u[9:10], min_idx
# end

function visualize!(vis, model::PlanarPushBlock, q, u; r = r,
        Δt = 0.1)

	default_background!(vis)

    setobject!(vis["box"], GeometryBasics.Rect(Vec(-1.0 * r,
		-1.0 * r,
		-1.0 * r),
		Vec(2.0 * r, 2.0 * r, 2.0 * r)),
		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

    for i = 1:model.nc
        setobject!(vis["contact$i"], GeometryBasics.Sphere(Point3f0(0),
            convert(Float32, 0.02)),
            MeshPhongMaterial(color = RGBA(1.0, 165.0 / 255.0, 0.0, 1.0)))
    end

	# for i = 1:4
    #     setobject!(vis["control$i"], GeometryBasics.Sphere(Point3f0(0),
    #         convert(Float32, 0.02)),
    #         MeshPhongMaterial(color = RGBA(51.0 / 255.0, 1.0, 1.0, 1.0)))
    # end

	# force_vis = ArrowVisualizer(vis[:force])
	# setobject!(force_vis, MeshPhongMaterial(color=RGBA(1.0, 0.0, 0.0, 1.0)))
	#
	# uf, up, min_idx = control_input(q[2], u[1])
	# uf_norm = uf / 10.0
	# settransform!(force_vis,
	# 			Point(up[1] - uf_norm[1], up[2] - uf_norm[2], 2 * r),
	# 			Vec(uf_norm[1], uf_norm[2], 2 * r),
	# 			shaft_radius=0.01,
	# 			max_head_radius=0.025)


    anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

	T = length(q)
    for t = 1:T-1
        MeshCat.atframe(anim, t) do
			if t < T-1
				# uf, up, min_idx = control_input(q[t+1], u[t])
				# println("t = $t, control_input: $min_idx")
				#
				# if norm(uf) < 1.0e-6
				# 	setvisible!(vis[:force], false)
				# else
				# 	setvisible!(vis[:force], true)
				# 	uf_norm = uf ./ 10.0
				# 	settransform!(force_vis,
				# 				Point(up[1] - uf_norm[1], up[2] - uf_norm[2], 2 * r),
				# 				Vec(uf_norm[1], uf_norm[2], 2 * r),
				# 				shaft_radius=0.01,
				# 				max_head_radius=0.025)
				# end
			end

            settransform!(vis["box"],
				compose(Translation(q[t+1][1], q[t+1][2], r), LinearMap(RotZ(q[t+1][3]))))

            for i = 1:model.nc
                settransform!(vis["contact$i"],
                    Translation(([q[t+1][1:2]; 0.0] + RotZ(q[t+1][3]) * [contact_corner_offset[i]; 0.0])...))
            end

			# for i = 1:4
			# 	settransform!(vis["control$i"],
			# 		Translation(([q[t+1][1:2]; r] + RotZ(q[t+1][3]) * [control_input_offset[i]; 0.0])...))
			# end
        end
    end

	settransform!(vis["/Cameras/default"],
		compose(Translation(0.0, 0.0, 50.0), LinearMap(RotZ(0.5 * pi) * RotY(-pi/2.5))))
	setprop!(vis["/Cameras/default/rotated/<object>"], "zoom", 25)


    MeshCat.setanimation!(vis, anim)
end
