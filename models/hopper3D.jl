"""
    hopper 3D
        orientation representation: modified rodrigues parameters
		similar to Raibert hopper, all mass is located at the body
		s = (px, py, pz, tx, ty, tz, r)
"""
struct Hopper3D{I, T} <: Model{I, T}
    n::Int
    m::Int
    d::Int

	mb # mass of body
    ml # mass of leg
    Jb # inertia of body
    Jl # inertia of leg

    μ  # coefficient of friction
    g  # gravity

    qL::Vector
    qU::Vector

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
nq = 7 # configuration dimension
nu = 3 # control dimension
nc = 1 # number of contact points
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

# Parameters
g = 9.81 # gravity
μ = 1.0  # coefficient of friction
mb = 1.0 # body mass
ml = 0.1  # leg mass
Jb = 0.25 # body inertia
Jl = 0.025 # leg inertia

# Kinematics
function kinematics(::Hopper3D, q)
	p = view(q, 1:3)
	R = MRP(view(q, 4:6)...)
	p + R*[0.0; 0.0; -1.0 * q[7]]
end

# Methods
function M_func(model::Hopper3D, q)
	Diagonal(@SVector [model.mb + model.ml, model.mb + model.ml, model.mb + model.ml,
					   model.Jb + model.Jl, model.Jb + model.Jl, model.Jb + model.Jl,
					   model.ml])
end

function C_func(model::Hopper3D, q, q̇)
	@SVector [0.0, 0.0, (model.mb + model.ml) * model.g, 0.0, 0.0, 0.0, 0.0]
end

function ϕ_func(model::Hopper3D, q)
    @SVector [kinematics(model, q)[3]]
end

function B_func(::Hopper3D, q)
    rot = view(q, 4:6)
    R = MRP(rot...)
    @SMatrix [0.0 0.0 0.0 R[1,1] R[2,1] R[3,1] 0.0;
              0.0 0.0 0.0 R[1,2] R[2,2] R[3,2] 0.0;
			  R[1,3] R[2,3] R[3,3] 0.0 0.0 0.0 1.0]
end

function N_func(model::Hopper3D, q)
	k(z) = [kinematics(model, z)[3]]
    ForwardDiff.jacobian(k, q)
end

function P_func(model::Hopper3D, q)
    map = [1.0 0.0;
           0.0 1.0;
           -1.0 0.0;
           0.0 -1.0]

    k(z) = kinematics(model, z)[1:2]
    map * ForwardDiff.jacobian(k, q)
end

function friction_cone(model::Hopper3D,u)
    λ = u[model.idx_λ]
    b = u[model.idx_b]

    @SVector [model.μ * λ[1] - sum(b[1:4])]
end

function maximum_dissipation(model::Hopper3D, x⁺, u, h)
    q3 = x⁺[model.nq .+ (1:model.nq)]
	q2 = x⁺[1:model.nq]

    ψ = u[model.idx_ψ]
    ψ_stack = ψ[1] * ones(4)
    η = u[model.idx_η]

    P_func(model, q3) * (q3 - q2) / h + ψ_stack - η
end

function lagrangian_derivatives(model, q, v)
	D1L = -1.0 * C_func(model, q, v)
    D2L = M_func(model, q) * v
	return D1L, D2L
end

function fd(model::Hopper3D{Discrete, FixedTime}, x⁺, x, u, w, h, t)
	q3 = view(x⁺, model.nq .+ (1:model.nq))
	q2⁺ = view(x⁺, 1:model.nq)
	q2⁻ = view(x, model.nq .+ (1:model.nq))
	q1 = view(x, 1:model.nq)
	u_ctrl = view(u, model.idx_u)
	λ = view(u, model.idx_λ)
	b = view(u, model.idx_b)

	# evalutate at midpoint
	qm1 = 0.5 * (q1 + q2⁺)
    vm1 = (q2⁺ - q1) / h
    qm2 = 0.5 * (q2⁺ + q3)
    vm2 = (q3 - q2⁺) / h

	D1L1, D2L1 = lagrangian_derivatives(model, qm1, vm1)
	D1L2, D2L2 = lagrangian_derivatives(model, qm2, vm2)

    [q2⁺ - q2⁻;
    (0.5 * h * D1L1 + D2L1 + 0.5 * h * D1L2 - D2L2
    + transpose(B_func(model, qm2)) * SVector{3}(u_ctrl)
    + transpose(N_func(model, q3)) * SVector{1}(λ)
    + transpose(P_func(model, q3)) * SVector{4}(b))]
end

function fd(model::Hopper3D{Discrete, FreeTime}, x⁺, x, u, w, h, t)
	q3 = view(x⁺, model.nq .+ (1:model.nq))
	q2⁺ = view(x⁺, 1:model.nq)
	q2⁻ = view(x, model.nq .+ (1:model.nq))
	q1 = view(x, 1:model.nq)
	u_ctrl = view(u, model.idx_u)
	λ = view(u, model.idx_λ)
	b = view(u, model.idx_b)
	h = u[end]

	# evalutate at midpoint
	qm1 = 0.5 * (q1 + q2⁺)
    vm1 = (q2⁺ - q1) / h
    qm2 = 0.5 * (q2⁺ + q3)
    vm2 = (q3 - q2⁺) / h

	D1L1, D2L1 = lagrangian_derivatives(model, qm1, vm1)
	D1L2, D2L2 = lagrangian_derivatives(model, qm2, vm2)

    [q2⁺ - q2⁻;
    (0.5 * h * D1L1 + D2L1 + 0.5 * h * D1L2 - D2L2
    + transpose(B_func(model, q3)) * SVector{3}(u_ctrl)
    + transpose(N_func(model, q3)) * SVector{1}(λ)
    + transpose(P_func(model, q3)) * SVector{4}(b))]
end

r = 0.5
qL = -Inf * ones(nq)
qU = Inf * ones(nq)
qL[3] = 0.0
qL[7] = 0.1
qU[7] = r

model = Hopper3D{Discrete, FixedTime}(n, m, d,
			mb, ml, Jb, Jl,
			μ, g,
			qL, qU,
            nq, nu, nc, nf, nb,
            idx_u,
            idx_λ,
            idx_b,
            idx_ψ,
            idx_η,
            idx_s)

function visualize!(vis, model::Hopper3D, q;
	Δt = 0.1)

	default_background!(vis)

	r_foot = 0.05
	setobject!(vis["body"], Sphere(Point3f0(0),
	   convert(Float32, 0.1)),
	   MeshPhongMaterial(color = RGBA(0, 1, 0, 1.0)))
	setobject!(vis["foot"], Sphere(Point3f0(0),
	   convert(Float32, r_foot)),
	   MeshPhongMaterial(color = RGBA(1.0, 165.0 / 255.0, 0.0, 1.0)))

	r_leg = 0.5 * r_foot
	n_leg = 100
	for i = 1:n_leg
	   setobject!(vis["leg$i"], Sphere(Point3f0(0),
	       convert(Float32, r_leg)),
	       MeshPhongMaterial(color = RGBA(0, 0, 0, 1.0)))
	end
	p_leg = [zeros(3) for i = 1:n_leg]
	anim = MeshCat.Animation(convert(Int,floor(1.0 / Δt)))

	z_shift = [0.0 ; 0.0; r_foot]
	for t = 1:length(q)
	   p_body = q[t][1:3]

	   q_tmp = Array(copy(q[t]))
	   r_range = range(0.0, stop = q[t][7], length = n_leg)
	   for i = 1:n_leg
	       q_tmp[7] = r_range[i]
	       p_leg[i] = kinematics(model, q_tmp)
	   end
	   q_tmp[7] = q[t][7]
	   p_foot = kinematics(model, q_tmp)

	   MeshCat.atframe(anim, t) do
	       settransform!(vis["body"], Translation(p_body + z_shift))
	       settransform!(vis["foot"], Translation(p_foot + z_shift))

	       for i = 1:n_leg
	           settransform!(vis["leg$i"], Translation(p_leg[i] + z_shift))
	       end
	   end
	end
	MeshCat.setanimation!(vis, anim)
end
