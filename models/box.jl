"""
    box
        particle with contacts at each corner
        orientation representation: modified rodrigues parameters
"""
struct Box{I, T} <: Model{I, T}
    n::Int
    m::Int
    d::Int

    mass
    J
    μ
    g

    r
	n_corners
    corner_offset

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
nq = 6 # configuration dimension
nu = 3 # control dimension
nc = 8 # number of contact points
nf = 4 # number of faces for friction cone pyramid
nb = nc*nf
ns = 1

n = 2*nq
m = nu + nc + nb + nc + nb + 1
d = 0

idx_u = (1:nu)
idx_λ = nu .+ (1:nc)
idx_b = nu + nc .+ (1:nb)
idx_ψ = nu + nc + nb .+ (1:nc)
idx_η = nu + nc + nb + nc .+ (1:nb)
idx_s = nu + nc + nb + nc + nb .+ (1:ns)

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

corner_offset = @SVector [c1, c2, c3, c4, c5, c6, c7, c8]

# Parameters
μ = 1.0  # coefficient of friction
g = 9.81
mass = 1.0   # mass
J = 1.0 / 12.0 * mass * ((2.0 * r)^2 + (2.0 * r)^2)

# Methods
M_func(model::Box, q) = Diagonal(@SVector [model.mass, model.mass, model.mass, model.J, model.J, model.J])

function C_func(model::Box, q, q̇)
	ω = q̇[4:6]
	SVector{6}([0., 0., model.mass * model.g, cross(ω, Diagonal(model.J * ones(3)) * ω)...])
end

function ϕ_func(model::Box, q)
    p = view(q, 1:3)
    rot = view(q, 4:6)

    R = MRP(rot...)

    @SVector [(p + R * model.corner_offset[1])[3],
              (p + R * model.corner_offset[2])[3],
              (p + R * model.corner_offset[3])[3],
              (p + R * model.corner_offset[4])[3],
              (p + R * model.corner_offset[5])[3],
              (p + R * model.corner_offset[6])[3],
              (p + R * model.corner_offset[7])[3],
              (p + R * model.corner_offset[8])[3]]
end

function B_func(::Box, q)
    rot = view(q, 4:6)
    R = MRP(rot...)
    @SMatrix [0. 0. 0. R[1,1] R[2,1] R[3,1];
              0. 0. 0. R[1,2] R[2,2] R[3,2];
              0. 0. 0. R[1,3] R[2,3] R[3,3]]
end

function N_func(model::Box, q)
    tmp(z) = ϕ_func(model, z)
    ForwardDiff.jacobian(tmp, q)
end

function P_func(model::Box, q)
    map = [1. 0.;
           0. 1.;
           -1. 0.;
           0. -1.]

    function p(x)
        pos = view(x, 1:3)
        rot = view(x, 4:6)

        R = MRP(rot...)

        [map * (pos + R * model.corner_offset[1])[1:2];
         map * (pos + R * model.corner_offset[2])[1:2];
         map * (pos + R * model.corner_offset[3])[1:2];
         map * (pos + R * model.corner_offset[4])[1:2];
         map * (pos + R * model.corner_offset[5])[1:2];
         map * (pos + R * model.corner_offset[6])[1:2];
         map * (pos + R * model.corner_offset[7])[1:2];
         map * (pos + R * model.corner_offset[8])[1:2]]
    end

    ForwardDiff.jacobian(p, q)
end

function friction_cone(model::Box,u)
    λ = u[model.idx_λ]
    b = u[model.idx_b]

    @SVector [model.μ * λ[1] - sum(b[1:4]),
              model.μ * λ[2] - sum(b[5:8]),
              model.μ * λ[3] - sum(b[9:12]),
              model.μ * λ[4] - sum(b[13:16]),
              model.μ * λ[5] - sum(b[17:20]),
              model.μ * λ[6] - sum(b[21:24]),
              model.μ * λ[7] - sum(b[25:28]),
              model.μ * λ[8] - sum(b[29:32])]
end

function maximum_dissipation(model::Box, x⁺, u, h)
    q3 = x⁺[model.nq .+ (1:model.nq)]
	q2 = x⁺[1:model.nq]

    ψ = u[model.idx_ψ]
    ψ_stack = [ψ[1] * ones(4);
               ψ[2] * ones(4);
               ψ[3] * ones(4);
               ψ[4] * ones(4);
               ψ[5] * ones(4);
               ψ[6] * ones(4);
               ψ[7] * ones(4);
               ψ[8] * ones(4)]

    η = u[model.idx_η]

    P_func(model, q3) * (q3 - q2) / h + ψ_stack - η
end

function lagrangian_derivatives(model, q, v)
	D1L = -1.0 * C_func(model, q, v)
    D2L = M_func(model, q) * v
	return D1L, D2L
end


function fd(model::Box{Discrete, FixedTime}, x⁺, x, u, w, h, t)
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
     + transpose(B_func(model, qm2)) * SVector{3}(u_ctrl)
     + transpose(N_func(model, q3)) * SVector{8}(λ)
     + transpose(P_func(model, q3)) * SVector{32}(b))]
end

model.nq
model.nu
model.nc
model.nb

model = Box{Discrete, FixedTime}(n, m, d,
			mass, J, μ, g,
            r, 8, corner_offset,
            nq, nu, nc, nf, nb,
            idx_u,
            idx_λ,
            idx_b,
            idx_ψ,
            idx_η,
            idx_s)


function visualize!(vis, model::Box, q;
        Δt = 0.1)

	default_background!(vis)

    setobject!(vis["box"], GeometryBasics.Rect(Vec(-1.0 * model.r,
		-1.0 * model.r,
		-1.0 * model.r),
		Vec(2.0 * model.r, 2.0 * model.r, 2.0 * model.r)))

    for i = 1:model.n_corners
        setobject!(vis["corner$i"], GeometryBasics.Sphere(Point3f0(0),
            convert(Float32, 0.05)),
            MeshPhongMaterial(color = RGBA(1.0, 165.0 / 255.0, 0.0, 1.0)))
    end

    anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

    for t = 1:length(q)
        MeshCat.atframe(anim, t) do

            settransform!(vis["box"],
				compose(Translation(q[t][1:3]...), LinearMap(MRP(q[t][4:6]...))))

            for i = 1:model.n_corners
                settransform!(vis["corner$i"],
                    Translation((q[t][1:3] + MRP(q[t][4:6]...) * (corner_offset[i]))...))
            end
        end
    end
    # settransform!(vis["/Cameras/default"], compose(Translation(-1, -1, 0),LinearMap(RotZ(pi/2))))
    MeshCat.setanimation!(vis, anim)
end
