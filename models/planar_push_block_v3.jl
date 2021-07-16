"""
    planar push block
        particle with contacts at each corner
"""
struct PlanarPushBlockV3{I, T} <: Model{I, T}
    n::Int
    m::Int
    d::Int

    mass_block
	mass_pusher

    J
    μ_surface
	μ_pusher
    g

    contact_corner_offset
	block_dim
	block_rnd

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

# signed distance for a box
function sd_box(p, dim)
	q = abs.(p) - dim
	norm(max.(q, 0.0)) + min(maximum(q), 0.0)
end

function sd_2d_box(p, pose, dim, rnd)
	x, y, θ = pose
	R = rotation_matrix(-θ)
	p_rot = R * (p - pose[1:2])

	return sd_box(p_rot, dim) - rnd
end

# Dimensions
nq = 5 # configuration dimension
nu = 2 # control dimension
nc = 5 # number of contact points
nf = 4 # number of faces for friction cone pyramid
nb = (nc - 1) * nf + 2
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
# cu1 = @SVector [r, r]
# cu2 = @SVector [-r, r]
# cu3 = @SVector [-r, -r]
# cu4 = @SVector [r, -r]
cu5 = @SVector [r, 0.0]
cu6 = @SVector [0.0, r]
cu7 = @SVector [-r, 0.0]
cu8 = @SVector [0.0, -r]

contact_corner_offset = @SVector [cc1, cc2, cc3, cc4]
# control_input_offset = @SVector [cu1, cu2, cu3, cu4, cu5, cu6, cu7, cu8]
control_input_offset = @SVector [cu5, cu6, cu7, cu8]

# Parameters
μ_surface = 1.0  # coefficient of friction
μ_pusher = 1.0
g = 9.81
mass_block = 1.0   # mass
mass_pusher = 0.1
J = 1.0 / 12.0 * mass_block * ((2.0 * r)^2 + (2.0 * r)^2)

rnd = 0.01
dim_rnd = [r - rnd, r - rnd]

# Methods
M_func(model::PlanarPushBlockV3, q) = Diagonal(@SVector [model.mass_block, model.mass_block,
	model.J, model.mass_pusher, model.mass_pusher])

function C_func(model::PlanarPushBlockV3, q, q̇)
	SVector{5}([0.0, 0.0, 0.0, 0.0, 0.0])
end

function rotation_matrix(x)
	SMatrix{2,2}([cos(x) -sin(x); sin(x) cos(x)])
end

function ϕ_func(model::PlanarPushBlockV3, q)
    p_block = view(q, 1:3)
	p_pusher = view(q, 4:5)

	sdf = sd_2d_box(p_pusher, p_block, model.block_dim, model.block_rnd)

    @SVector [0.0, 0.0, 0.0, 0.0, sdf]
end

function B_func(model::PlanarPushBlockV3, q)
	SMatrix{5,2}([0.0 0.0;
				  0.0 0.0;
				  0.0 0.0;
				  1.0 0.0;
				  0.0 1.0])
end

function N_func(model::PlanarPushBlockV3, q)
    tmp(z) = ϕ_func(model, z)
    ForwardDiff.jacobian(tmp, q)
end

function P_func(model::PlanarPushBlockV3, q)
	map1 = [1.0;
	        -1.0]

    map2 = [1. 0.;
           0. 1.;
           -1. 0.;
           0. -1.]



    function p(x)
        pos = view(x, 1:2)
		θ = x[3]
        R = rotation_matrix(θ)

        [map2 * (pos + R * model.contact_corner_offset[1])[1:2];
         map2 * (pos + R * model.contact_corner_offset[2])[1:2];
         map2 * (pos + R * model.contact_corner_offset[3])[1:2];
         map2 * (pos + R * model.contact_corner_offset[4])[1:2]]
    end

    P_block = ForwardDiff.jacobian(p, q)

	# pusher block
	p_block = view(q, 1:3)
	p_pusher = view(q, 4:5)

	sd_p(x) = sd_2d_box(x, p_block, model.block_dim, model.block_rnd)
	Np = ForwardDiff.gradient(sd_p, p_pusher)

	n_dir = Np[1:2] ./ norm(Np[1:2])
	t_dir = rotation_matrix(0.5 * π) * n_dir

	r = p_pusher - p_block[1:2]
	m = cross([r; 0.0], [t_dir; 0.0])[3]

	P_pusherblock = map1 * [t_dir[1]; t_dir[2]; cross([p_pusher; 0.0] - [p_block[1:2]; 0.0], [t_dir; 0.0])[3]; -t_dir[1]; -t_dir[2]]'

	return [P_block; P_pusherblock]
end

function friction_cone(model::PlanarPushBlockV3,u)
    λ = u[model.idx_λ]
    b = u[model.idx_b]

    @SVector [model.μ_surface[1] * λ[1] - sum(b[1:4]),
              model.μ_surface[2] * λ[2] - sum(b[5:8]),
              model.μ_surface[3] * λ[3] - sum(b[9:12]),
              model.μ_surface[4] * λ[4] - sum(b[13:16]),
			  model.μ_pusher * λ[5] - sum(b[17:18])]
end

function maximum_dissipation(model::PlanarPushBlockV3, x⁺, u, h)
    q3 = x⁺[model.nq .+ (1:model.nq)]
	q2 = x⁺[1:model.nq]

    ψ = u[model.idx_ψ]
    ψ_stack = [ψ[1] * ones(4);
               ψ[2] * ones(4);
               ψ[3] * ones(4);
               ψ[4] * ones(4);
			   ψ[5] * ones(2)]

    η = u[model.idx_η]

    P_func(model, q3) * (q3 - q2) / h + ψ_stack - η
end

function lagrangian_derivatives(model, q, v)
	D1L = -1.0 * C_func(model, q, v)
    D2L = M_func(model, q) * v
	return D1L, D2L
end


function fd(model::PlanarPushBlockV3{Discrete, FixedTime}, x⁺, x, u, w, h, t)
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
     + B_func(model, qm2) * SVector{2}(u_ctrl[1:2])
     + transpose(N_func(model, q3)) * SVector{5}(λ)
     + transpose(P_func(model, q3)) * SVector{18}(b))]
end

model = PlanarPushBlockV3{Discrete, FixedTime}(n, m, d,
			mass_block, mass_pusher, J,
			[μ_surface for i = 1:nc], μ_pusher,
			g,
			contact_corner_offset,
			dim_rnd,
			rnd,
            nq, nu, nc, nf, nb,
            idx_u,
            idx_λ,
            idx_b,
            idx_ψ,
            idx_η,
            idx_s)


function visualize!(vis, model::PlanarPushBlockV3, q, u; r = r,
        Δt = 0.1)

	default_background!(vis)

    setobject!(vis["box"], GeometryBasics.Rect(Vec(-1.0 * r,
		-1.0 * r,
		-1.0 * r),
		Vec(2.0 * r, 2.0 * r, 2.0 * r)),
		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

    for i = 1:4
        setobject!(vis["contact$i"], GeometryBasics.Sphere(Point3f0(0),
            convert(Float32, 0.02)),
            MeshPhongMaterial(color = RGBA(1.0, 165.0 / 255.0, 0.0, 1.0)))
    end

	force_vis = ArrowVisualizer(vis[:force])
	setobject!(force_vis, MeshPhongMaterial(color=RGBA(1.0, 0.0, 0.0, 1.0)))

	us = u[1] / 10.0

	settransform!(force_vis,
				Point(q[1][4] - us[1], q[1][5] - us[1], 2 * r),
				Vec(us[1], us[1], 2 * r),
				shaft_radius=0.01,
				max_head_radius=0.025)


    anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

	T = length(q)
    for t = 1:T-1
        MeshCat.atframe(anim, t) do
			if t < T-1

				if norm(u[t]) < 1.0e-6
					setvisible!(vis[:force], false)
				else
					setvisible!(vis[:force], true)

					us = u[t] / 10.0
					settransform!(force_vis,
								Point(q[t+1][4] - us[1], q[t+1][5] - us[2], 2 * r),
								Vec(us[1], us[2], 2 * r),
								shaft_radius=0.01,
								max_head_radius=0.025)
				end
			end

            settransform!(vis["box"],
				compose(Translation(q[t+1][1], q[t+1][2], r), LinearMap(RotZ(q[t+1][3]))))

            for i = 1:4
                settransform!(vis["contact$i"],
                    Translation(([q[t+1][1:2]; 0.0] + RotZ(q[t+1][3]) * [contact_corner_offset[i]; 0.0])...))
            end
        end
    end

	settransform!(vis["/Cameras/default"],
		compose(Translation(0.0, 0.0, 50.0), LinearMap(RotZ(0.5 * pi) * RotY(-pi/2.5))))
	setprop!(vis["/Cameras/default/rotated/<object>"], "zoom", 25)


    MeshCat.setanimation!(vis, anim)
end
