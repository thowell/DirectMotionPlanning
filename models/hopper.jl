"""
    Hopper
    	model inspired by "Dynamically Stable Legged Locomotion"
		s = (x, z, t, r)
			x - lateral position
			z - vertical position
			t - body orientation
			r - leg length
"""
struct Hopper{I, T} <: Model{I, T}
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

	uL::Vector
	uU::Vector

    nq
    nu
    nc
    nf
    nb
	ns

    idx_u
    idx_λ
    idx_b
    idx_ψ
    idx_η
    idx_s
end

# Dimensions
nq = 4 # configuration dimension
nu = 2 # control dimension
nc = 1 # number of contact points
nf = 2 # number of faces for friction cone
nb = nc * nf
ns = 1

# Parameters
g = 9.81 # gravity
μ = 1.0  # coefficient of friction
mb = 1.0 # body mass
ml = 0.1  # leg mass
Jb = 0.25 # body inertia
Jl = 0.025 # leg inertia

n = 2 * nq
m = nu + nc + nb + nc + nb + ns
d = nq

idx_u = (1:nu)
idx_λ = nu .+ (1:nc)
idx_b = nu + nc .+ (1:nb)
idx_ψ = nu + nc + nb .+ (1:nc)
idx_η = nu + nc + nb + nc .+ (1:nb)
idx_s = nu + nc + nb + nc + nb .+ (1:ns)

# Kinematics
kinematics(::Hopper, q) = [q[1] + q[4] * sin(q[3]), q[2] - q[4] * cos(q[3])]

# Methods
M_func(model::Hopper, q) = Diagonal(@SVector [
								 model.mb + model.ml,
								 model.mb + model.ml,
								 model.Jb + model.Jl,
								 model.ml])

C_func(model::Hopper, q, q̇) = @SVector [0.0,
									 (model.mb + model.ml) * model.g,
									 0.0,
									 0.0]

function ϕ_func(::Hopper, q)
    @SVector [q[2] - q[4] * cos(q[3])]
end

N_func(::Hopper, q) = @SMatrix [0.0 1.0 (q[4] * sin(q[3])) (-1.0 * cos(q[3]))]

function _P_func(model, q)
	@SMatrix [1.0 0.0 (q[4] * cos(q[3])) sin(q[3])]
end

function P_func(::Hopper, q)
    @SMatrix [1.0 0.0 (q[4] * cos(q[3])) sin(q[3]);
    	-1.0 0.0 (-1.0 * q[4] * cos(q[3])) -1.0 * sin(q[3])]
end

B_func(::Hopper, q) = @SMatrix [0.0 0.0 1.0 0.0;
                                -sin(q[3]) cos(q[3]) 0.0 1.0]

function lagrangian_derivatives(model, q, v)
	D1L = -1.0 * C_func(model, q, v)
    D2L = M_func(model, q) * v
	return D1L, D2L
end

function fd(model::Hopper{Discrete, FixedTime}, x⁺, x, u, w, h, t)
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
    + (transpose(B_func(model, qm2)) * SVector{2}(u_ctrl)
    + transpose(N_func(model, q3)) * SVector{1}(λ)
    + transpose(P_func(model, q3)) * SVector{2}(b))
    + h * w)]
end

function maximum_dissipation(model::Hopper{Discrete, FixedTime}, x⁺, u, h)
	q3 = x⁺[model.nq .+ (1:model.nq)]
	q2 = x⁺[1:model.nq]
	ψ = u[model.idx_ψ]
	ψ_stack = ψ[1] * ones(model.nf)
	η = u[model.idx_η]
	return P_func(model, q3) * (q3 - q2) / h + ψ_stack - η
end

function fd(model::Hopper{Discrete, FreeTime}, x⁺, x, u, w, h, t)
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
	+ (transpose(B_func(model, qm2)) * SVector{2}(u_ctrl)
	+ transpose(N_func(model, q3)) * SVector{1}(λ)
	+ transpose(P_func(model, q3)) * SVector{2}(b))
	+ h * w)]
end

function maximum_dissipation(model::Hopper{Discrete, FreeTime}, x⁺, u, h)
	q3 = x⁺[model.nq .+ (1:model.nq)]
	q2 = x⁺[1:model.nq]
	ψ = u[model.idx_ψ]
	ψ_stack = ψ[1] * ones(model.nf)
	η = u[model.idx_η]
	h = u[end]
	return P_func(model, q3) * (q3 - q2) / h + ψ_stack - η
end

function no_slip(model::Hopper{Discrete, FixedTime}, x⁺, u, h)
	q3 = view(x⁺, model.nq .+ (1:model.nq))
	q2 = view(x⁺, 1:model.nq)
	λ = view(u, model.idx_λ)
	s = view(u, model.idx_s)

	return s[1] - (λ' * _P_func(model, q3) * (q3 - q2) / h)[1]
end

function no_slip(model::Hopper{Discrete, FreeTime}, x⁺, u, h)
	q3 = view(x⁺, model.nq .+ (1:model.nq))
	q2 = view(x⁺, 1:model.nq)
	λ = view(u, model.idx_λ)
	s = view(u, model.idx_s)
	h = u[end]
	return s[1] - (λ' * _P_func(model, q3) * (q3 - q2) / h)[1]
end

function friction_cone(model::Hopper, u)
	λ = u[model.idx_λ]
	b = u[model.idx_b]
	return @SVector [model.μ * λ[1] - sum(b)]
end

r = 0.5
qL = -Inf * ones(nq)
qU = Inf * ones(nq)
qL[4] = 0.1
qU[4] = r

uL = -100.0 * ones(nu)
uU = 100.0 * ones(nu)

model = Hopper{Discrete, FixedTime}(n, m, d,
			   mb, ml, Jb, Jl,
			   μ, g,
			   qL, qU,
			   uL, uU,
			   nq,
		       nu,
		       nc,
		       nf,
		       nb,
		   	   ns,
		       idx_u,
		       idx_λ,
		       idx_b,
		       idx_ψ,
		       idx_η,
		       idx_s)

# Visualization
function visualize!(vis, model::Hopper, q;
		Δt = 0.1, scenario = :vertical)

    r_foot = 0.05
    r_leg = 0.5 * r_foot

	default_background!(vis)

    setobject!(vis["body"], Sphere(Point3f0(0),
        convert(Float32, 0.1)),
        MeshPhongMaterial(color = RGBA(0, 1, 0, 1.0)))

    setobject!(vis["foot"], Sphere(Point3f0(0),
        convert(Float32, r_foot)),
        MeshPhongMaterial(color = RGBA(1.0, 165.0 / 255.0, 0, 1.0)))

    n_leg = 100
    for i = 1:n_leg
        setobject!(vis["leg$i"], Sphere(Point3f0(0),
            convert(Float32, r_leg)),
            MeshPhongMaterial(color = RGBA(0, 0, 0, 1.0)))
    end

    p_leg = [zeros(3) for i = 1:n_leg]
    anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

    for t = 1:length(q)
        p_body = [q[t][1], 0.0, q[t][2]]
        p_foot = [kinematics(model, q[t])[1], 0.0, kinematics(model, q[t])[2]]

        q_tmp = Array(copy(q[t]))
        r_range = range(0, stop = q[t][4], length = n_leg)
        for i = 1:n_leg
            q_tmp[4] = r_range[i]
            p_leg[i] = [kinematics(model, q_tmp)[1], 0.0, kinematics(model, q_tmp)[2]]
        end
        q_tmp[4] = q[t][4]
        p_foot = [kinematics(model, q_tmp)[1], 0.0, kinematics(model, q_tmp)[2]]

        z_shift = [0.0; 0.0; r_foot]

        MeshCat.atframe(anim, t) do
            settransform!(vis["body"], Translation(p_body + z_shift))
            settransform!(vis["foot"], Translation(p_foot + z_shift))

            for i = 1:n_leg
                settransform!(vis["leg$i"], Translation(p_leg[i] + z_shift))
            end
        end
    end

	settransform!(vis["/Cameras/default"],
		compose(Translation(0.0, 0.5, -1.0),LinearMap(RotZ(-pi / 2.0))))

    MeshCat.setanimation!(vis, anim)
end


qq = ones(nq)

M_func(model, qq)
B_func(model, qq)
