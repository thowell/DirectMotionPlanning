"""
    HopperImpedance
		s = (x, z, t, r)
			x - lateral position
			z - vertical position
			t - body orientation
			r - leg length
"""

struct HopperImpedance{I, T} <: Model{I, T}
    n::Int
    m::Int
    d::Int

    mb # mass of body
	Jb # inertia of body

    mf # mass of foot

    μ  # coefficient of friction
	k0 # spring constant
	r0 # leg equilibrium length
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
nu = 3 # control dimension
nc = 2 # number of contact points
nf = 2 # number of faces for friction cone
nb = nc * nf
ns = 1

# Parameters
g = 9.81 # gravity
μ = 1.0  # coefficient of friction
mb = 1.0 # body mass
mf = 0.01  # leg mass
Jb = 0.1 # body inertia
k0 = 500.0 # 375.0
r0 = 0.5

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
kinematics(::HopperImpedance, q) = [q[1], q[2],
	q[1] + q[4] * sin(q[3]), q[2] - q[4] * cos(q[3])]

function jacobian(::HopperImpedance, q)
	@SMatrix [1.0 0.0 0.0 0.0;
	          0.0 1.0 0.0 0.0;
			  1.0 0.0 q[4] * cos(q[3]) sin(q[3]);
	          0.0 1.0 q[4] * sin(q[3]) -1.0 * cos(q[3])]
end

# Lagrangian

function lagrangian(model::HopperImpedance, q, q̇)
	L = 0.0

	# body
	L += 0.5 * model.mb * (q̇[1]^2.0 + q̇[2]^2.0)
	L += 0.5 * model.Jb * q̇[3]^2.0
	L -= model.mb * model.g * q[2]

	# foot
	v_foot = jacobian(model, q) * q̇
	L += 0.5 * model.mf * v_foot' * v_foot
	L -= model.mf * model.g * kinematics(model, q)[2]

	# spring
	# L -= 0.5 * model.k0 * (q[4] - model.r0)^2.0 # this is included in the control input

	return L
end

function dLdq(model, q, q̇)
	Lq(x) = lagrangian(model, x, q̇)
	ForwardDiff.gradient(Lq, q)
end

function dLdq̇(model, q, q̇)
	Lq̇(x) = lagrangian(model, q, x)
	ForwardDiff.gradient(Lq̇, q̇)
end

function C_func(model, q, q̇)
	tmp_q(z) = dLdq̇(model, z, q̇)
	tmp_q̇(z) = dLdq̇(model, q, z)

	ForwardDiff.jacobian(tmp_q, q) * q̇ - dLdq(model, q, q̇)
end

# Methods
function M_func(model::HopperImpedance, q)
	M = Diagonal([model.mb, model.mb, model.Jb, 0.0])

	J_foot = jacobian(model, q)
	M += model.mf * J_foot' * J_foot

	return M
end

function ϕ_func(::HopperImpedance, q)
    @SVector [q[2], q[2] - q[4] * cos(q[3])]
end

function N_func(::HopperImpedance, q)
	@SMatrix [0.0 1.0 0.0 0.0;
	          0.0 1.0 (q[4] * sin(q[3])) (-1.0 * cos(q[3]))]
end

function _P_func(model, q)
	@SMatrix [1.0 0.0 0.0 0.0;
			  1.0 0.0 (q[4] * cos(q[3])) sin(q[3])]
end

function P_func(::HopperImpedance, q)
    @SMatrix [1.0 0.0 0.0 0.0;
			   1.0 0.0 (q[4] * cos(q[3])) sin(q[3]);
			  -1.0 0.0 0.0 0.0;
              -1.0 0.0 (-1.0 * q[4] * cos(q[3])) -1.0 * sin(q[3])]
end

function B_func(::HopperImpedance, q)
	# Diagonal(@SVector ones(4))
	@SMatrix [0.0 0.0 1.0 0.0;
              -sin(q[3]) cos(q[3]) 0.0 1.0;
			  0.0 0.0 0.0 (q[4] - model.r0)]
 end

function fd(model::HopperImpedance{Discrete, FixedTime}, x⁺, x, u, w, h, t)
	q3 = view(x⁺, model.nq .+ (1:model.nq))
	q2⁺ = view(x⁺, 1:model.nq)
	q2⁻ = view(x, model.nq .+ (1:model.nq))
	q1 = view(x, 1:model.nq)
	u_ctrl = view(u, model.idx_u)
	λ = view(u, model.idx_λ)
	b = view(u, model.idx_b)

    [q2⁺ - q2⁻;
	((1.0 / h) * (M_func(model, q1) * (SVector{4}(q2⁺) - SVector{4}(q1))
    - M_func(model, q2⁺) * (SVector{4}(q3) - SVector{4}(q2⁺)))
    + h * (transpose(B_func(model, q3)) * SVector{3}(u_ctrl)
    + transpose(N_func(model, q3)) * SVector{2}(λ)
    + transpose(P_func(model, q3)) * SVector{4}(b)
	- C_func(model, q3, (q3 - q2⁺) / h)
    + w))]
end

function maximum_dissipation(model::HopperImpedance{Discrete, FixedTime}, x⁺, u, h)
	q3 = x⁺[model.nq .+ (1:model.nq)]
	q2 = x⁺[1:model.nq]
	ψ = u[model.idx_ψ]
	ψ_stack = [ψ[1] * ones(model.nf); ψ[2] * ones(model.nf)]
	η = u[model.idx_η]
	return P_func(model, q3) * (q3 - q2) / h + ψ_stack - η
end

function fd(model::HopperImpedance{Discrete, FreeTime}, x⁺, x, u, w, h, t)
	q3 = view(x⁺, model.nq .+ (1:model.nq))
	q2⁺ = view(x⁺, 1:model.nq)
	q2⁻ = view(x, model.nq .+ (1:model.nq))
	q1 = view(x, 1:model.nq)
	u_ctrl = view(u, model.idx_u)
	λ = view(u, model.idx_λ)
	b = view(u, model.idx_b)
	h = u[end]

	[q2⁺ - q2⁻;
	((1.0 / h) * (M_func(model, q1) * (SVector{4}(q2⁺) - SVector{4}(q1))
	- M_func(model, q2⁺) * (SVector{4}(q3) - SVector{4}(q2⁺)))
	+ h * (transpose(B_func(model, q3)) * SVector{3}(u_ctrl)
	+ transpose(N_func(model, q3)) * SVector{2}(λ)
	+ transpose(P_func(model, q3)) * SVector{4}(b)
	- C_func(model, q3, (q3 - q2⁺) / h)
	+ w))]
end

function maximum_dissipation(model::HopperImpedance{Discrete, FreeTime}, x⁺, u, h)
	q3 = x⁺[model.nq .+ (1:model.nq)]
	q2 = x⁺[1:model.nq]
	ψ = u[model.idx_ψ]
	ψ_stack = [ψ[1] * ones(model.nf); ψ[2] * ones(model.nf)]
	η = u[model.idx_η]
	h = u[end]
	return P_func(model, q3) * (q3 - q2) / h + ψ_stack - η
end

function no_slip(model::HopperImpedance{Discrete, FixedTime}, x⁺, u, h)
	q3 = view(x⁺, model.nq .+ (1:model.nq))
	q2 = view(x⁺, 1:model.nq)
	λ = view(u, model.idx_λ)
	s = view(u, model.idx_s)

	return s[1] - (λ' * _P_func(model, q3) * (q3 - q2) / h)[1]
end

function no_slip(model::HopperImpedance{Discrete, FreeTime}, x⁺, u, h)
	q3 = view(x⁺, model.nq .+ (1:model.nq))
	q2 = view(x⁺, 1:model.nq)
	λ = view(u, model.idx_λ)
	s = view(u, model.idx_s)
	h = u[end]
	return s[1] - (λ' * _P_func(model, q3) * (q3 - q2) / h)[1]
end

function friction_cone(model::HopperImpedance, u)
	λ = u[model.idx_λ]
	b = u[model.idx_b]
	return @SVector [model.μ * λ[1] - sum(b[1:model.nf]),
	                 model.μ * λ[2] - sum(b[model.nf .+ (1:model.nf)])]
end

qL = -Inf * ones(nq)
qU = Inf * ones(nq)
qL[4] = 0.25 * r0
qU[4] = 1.25 * r0

uL = [-100.0; -100.0; 0.0]
uU = [100.0; 100.0; 1000.0]

model = HopperImpedance{Discrete, FixedTime}(n, m, d,
			   mb, Jb, mf,
			   μ, k0, r0, g,
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
function visualize!(vis, model::HopperImpedance, q;
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
       p_foot = [kinematics(model, q[t])[3], 0.0, kinematics(model, q[t])[4]]

       q_tmp = Array(copy(q[t]))
       r_range = range(0, stop = q[t][4], length = n_leg)
       for i = 1:n_leg
           q_tmp[4] = r_range[i]
           p_leg[i] = [kinematics(model, q_tmp)[3], 0.0, kinematics(model, q_tmp)[4]]
       end
       q_tmp[4] = q[t][4]
       p_foot = [kinematics(model, q_tmp)[3], 0.0, kinematics(model, q_tmp)[4]]

       z_shift = [0.0; 0.0; r_foot]

       MeshCat.atframe(anim, t) do
           settransform!(vis["body"], Translation(p_body + z_shift))
           settransform!(vis["foot"], Translation(p_foot + z_shift))

           for i = 1:n_leg
               settransform!(vis["leg$i"], Translation(p_leg[i] + z_shift))
           end
       end
   end

	if scenario == :flip
		settransform!(vis["/Cameras/default"],
			compose(Translation(0.0, 0.5, -1.0),LinearMap(RotZ(-pi / 2.0))))
	elseif scenario == :vertical
		settransform!(vis["/Cameras/default"],
			compose(Translation(0.0, 0.5, -1.0),LinearMap(RotZ(-pi / 2.0))))
	end

   MeshCat.setanimation!(vis, anim)
end
