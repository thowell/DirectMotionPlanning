struct Biped{I, T} <: Model{I, T}
    n::Int
    m::Int
    d::Int

    g
    μ

    # torso
    l_torso
    d_torso
    m_torso
    J_torso

    # leg 1
        # thigh
    l_thigh1
    d_thigh1
    m_thigh1
    J_thigh1

        # calf
    l_calf1
    d_calf1
    m_calf1
    J_calf1

    # leg 2
        # thigh
    l_thigh2
    d_thigh2
    m_thigh2
    J_thigh2

        # calf
    l_calf2
    d_calf2
    m_calf2
    J_calf2

    # joint limits
    qL
    qU

    # torque limits
    uL
    uU

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
nq = 2 + 5                # configuration dimension
nu = 4                    # control dimension
nc = 2                    # number of contact points
nf = 2                    # number of parameters for friction cone
nb = nc * nf
ns = 1

# World parameters
μ = 1.0      # coefficient of friction
g = 9.81     # gravity

# Model parameters
m_torso = 0.5 + 0.48 * 2.0
m_thigh = 0.8112
m_calf = 0.3037

J_torso = 0.0029
J_thigh = 0.00709
J_calf = 0.00398

l_torso = 0.15 + 0.15
l_thigh = 0.2755
l_calf = 0.308

d_torso = 0.0342
d_thigh = 0.2176
d_calf = 0.1445

n = 2 * nq
m = nu + nc + nb + nc + nb + ns
d = nq

idx_u = (1:nu)
idx_λ = nu .+ (1:nc)
idx_b = nu + nc .+ (1:nb)
idx_ψ = nu + nc + nb .+ (1:nc)
idx_η = nu + nc + nb + nc .+ (1:nb)
idx_s = nu + nc + nb + nc + nb .+ (1:ns)

qL = -Inf * ones(nq)
qU = Inf * ones(nq)

uL = -100.0 * ones(nu)
uU = 100.0 * ones(nu)

function kinematics_1(model::Biped, q; body = :torso, mode = :ee)
	x = q[1]
	z = q[2]

	if body == :torso
		l = model.l_torso
		d = model.d_torso
		θ = q[3]
		if mode == :ee
			return [x - l * sin(θ); z + l * cos(θ)]
		elseif mode == :com
			return [x - d * sin(θ); z + d * cos(θ)]
		end
	elseif body == :thigh_1
		l = model.l_thigh1
		d = model.d_thigh1
		θ = q[4]
	elseif body == :thigh_2
		l = model.l_thigh2
		d = model.d_thigh2
		θ = q[6]
	else
		@error "incorrect body specification"
	end

	if mode == :ee
		return [x + l * sin(θ); z - l * cos(θ)]
	elseif mode == :com
		return [x + d * sin(θ); z - d * cos(θ)]
	else
		@error "incorrect mode specification"
	end
end

function jacobian_1(model::Biped, q; body = :torso, mode = :ee)
	jac = zeros(eltype(q), 2, model.nq)
	jac[1, 1] = 1.0
	jac[2, 2] = 1.0
	if body == :torso
		r = mode == :ee ? model.l_torso : model.d_torso
		θ = q[3]
		jac[1, 3] = -r * cos(θ)
		jac[2, 3] = -r * sin(θ)
	elseif body == :thigh_1
		r = mode == :ee ? model.l_thigh1 : model.d_thigh1
		θ = q[4]
		jac[1, 4] = r * cos(θ)
		jac[2, 4] = r * sin(θ)
	elseif body == :thigh_2
		r = mode == :ee ? model.l_thigh2 : model.d_thigh2
		θ = q[6]
		jac[1, 6] = r * cos(θ)
		jac[2, 6] = r * sin(θ)
	else
		@error "incorrect body specification"
	end

	return jac
end

function kinematics_2(model::Biped, q; body = :calf_1, mode = :ee)

	if body == :calf_1
		p = kinematics_1(model, q, body = :thigh_1, mode = :ee)

		θb = q[5]

		lb = model.l_calf1
		db = model.d_calf1
	elseif body == :calf_2
		p = kinematics_1(model, q, body = :thigh_2, mode = :ee)

		θb = q[7]

		lb = model.l_calf2
		db = model.d_calf2
	else
		@error "incorrect body specification"
	end

	if mode == :ee
		return p + [lb * sin(θb); -1.0 * lb * cos(θb)]
	elseif mode == :com
		return p + [db * sin(θb); -1.0 * db * cos(θb)]
	else
		@error "incorrect mode specification"
	end
end

function jacobian_2(model::Biped, q; body = :calf_1, mode = :ee)

	if body == :calf_1
		jac = jacobian_1(model, q, body = :thigh_1, mode = :ee)

		θb = q[5]

		r = mode == :ee ? model.l_calf1 : model.d_calf1

		jac[1, 5] += r * cos(θb)
		jac[2, 5] += r * sin(θb)
	elseif body == :calf_2
		jac = jacobian_1(model, q, body = :thigh_2, mode = :ee)

		θb = q[7]

		r = mode == :ee ? model.l_calf2 : model.d_calf2

		jac[1, 7] += r * cos(θb)
		jac[2, 7] += r * sin(θb)
	else
		@error "incorrect body specification"
	end

	return jac
end

# Lagrangian

function lagrangian(model::Biped, q, q̇)
	L = 0.0

	# torso
	p_torso = kinematics_1(model, q, body = :torso, mode = :com)
	J_torso = jacobian_1(model, q, body = :torso, mode = :com)
	v_torso = J_torso * q̇

	L += 0.5 * model.m_torso * transpose(v_torso) * v_torso
	L += 0.5 * model.J_torso * q̇[3]^2.0
	L -= model.m_torso * model.g * p_torso[2]

	# thigh 1
	p_thigh_1 = kinematics_1(model, q, body = :thigh_1, mode = :com)
	J_thigh_1 = jacobian_1(model, q, body = :thigh_1, mode = :com)
	v_thigh_1 = J_thigh_1 * q̇

	L += 0.5 * model.m_thigh1 * transpose(v_thigh_1) * v_thigh_1
	L += 0.5 * model.J_thigh1 * q̇[4]^2.0
	L -= model.m_thigh1 * model.g * p_thigh_1[2]

	# leg 1
	p_calf_1 = kinematics_2(model, q, body = :calf_1, mode = :com)
	J_calf_1 = jacobian_2(model, q, body = :calf_1, mode = :com)
	v_calf_1 = J_calf_1 * q̇

	L += 0.5 * model.m_calf1 * transpose(v_calf_1) * v_calf_1
	L += 0.5 * model.J_calf1 * q̇[5]^2.0
	L -= model.m_calf1 * model.g * p_calf_1[2]

	# thigh 2
	p_thigh_2 = kinematics_1(model, q, body = :thigh_2, mode = :com)
	J_thigh_2 = jacobian_1(model, q, body = :thigh_2, mode = :com)
	v_thigh_2 = J_thigh_2 * q̇

	L += 0.5 * model.m_thigh2 * transpose(v_thigh_2) * v_thigh_2
	L += 0.5 * model.J_thigh2 * q̇[6]^2.0
	L -= model.m_thigh2 * model.g * p_thigh_2[2]

	# leg 2
	p_calf_2 = kinematics_2(model, q, body = :calf_2, mode = :com)
	J_calf_2 = jacobian_2(model, q, body = :calf_2, mode = :com)
	v_calf_2 = J_calf_2 * q̇

	L += 0.5 * model.m_calf2 * transpose(v_calf_2) * v_calf_2
	L += 0.5 * model.J_calf2 * q̇[7]^2.0
	L -= model.m_calf2 * model.g * p_calf_2[2]

	return L
end

function _dLdq(model::Biped, q, q̇)
	Lq(x) = lagrangian(model, x, q̇)
	ForwardDiff.gradient(Lq, q)
end

function _dLdq̇(model::Biped, q, q̇)
	Lq̇(x) = lagrangian(model, q, x)
	ForwardDiff.gradient(Lq̇, q̇)
end

# Methods
function M_func(model::Biped, q)
	M = Diagonal([0.0, 0.0, model.J_torso, model.J_thigh1, model.J_calf1, model.J_thigh2, model.J_calf2])

	# torso
	J_torso = jacobian_1(model, q, body = :torso, mode = :com)
	M += model.m_torso * J_torso' * J_torso

	# thigh 1
	J_thigh_1 = jacobian_1(model, q, body = :thigh_1, mode = :com)
	M += model.m_thigh1 * J_thigh_1' * J_thigh_1

	# leg 1
	J_calf_1 = jacobian_2(model, q, body = :calf_1, mode = :com)
	M += model.m_calf1 * J_calf_1' * J_calf_1

	# thigh 2
	J_thigh_2 = jacobian_1(model, q, body = :thigh_2, mode = :com)
	M += model.m_thigh2 * J_thigh_2' * J_thigh_2

	# leg 2
	J_calf_2 = jacobian_2(model, q, body = :calf_2, mode = :com)
	M += model.m_calf2 * J_calf_2' * J_calf_2

	return M
end

function _C_func(model::Biped, q, q̇)
	tmp_q(z) = _dLdq̇(model, z, q̇)
	tmp_q̇(z) = _dLdq̇(model, q, z)

	ForwardDiff.jacobian(tmp_q, q) * q̇ - _dLdq(model, q, q̇)
end

function ϕ_func(model::Biped, q)
	p_calf_1 = kinematics_2(model, q, body = :calf_1, mode = :ee)
	p_calf_2 = kinematics_2(model, q, body = :calf_2, mode = :ee)

	@SVector [p_calf_1[2], p_calf_2[2]]
end

function B_func(model::Biped, q)
	@SMatrix [0.0 0.0 0.0 1.0 0.0 0.0 0.0;
			  0.0 0.0 0.0 0.0 1.0 0.0 0.0;
			  0.0 0.0 0.0 0.0 0.0 1.0 0.0;
			  0.0 0.0 0.0 0.0 0.0 0.0 1.0]
end

function N_func(model::Biped, q)
	J_calf_1 = jacobian_2(model, q, body = :calf_1, mode = :ee)
	J_calf_2 = jacobian_2(model, q, body = :calf_2, mode = :ee)

	return [view(J_calf_1, 2:2, :);
			view(J_calf_2, 2:2, :)]
end

function _P_func(model::Biped, q)
	J_calf_1 = jacobian_2(model, q, body = :calf_1, mode = :ee)
	J_calf_2 = jacobian_2(model, q, body = :calf_2, mode = :ee)

	return [view(J_calf_1, 1:1, :);
			view(J_calf_2, 1:1, :)]
end

function P_func(model::Biped, q)
	J_calf_1 = jacobian_2(model, q, body = :calf_1, mode = :ee)
	J_calf_2 = jacobian_2(model, q, body = :calf_2, mode = :ee)
	map = [1.0; -1.0]

	return [map * view(J_calf_1, 1:1, :);
			map * view(J_calf_2, 1:1, :)]
end

function friction_cone(model::Biped, u)
	λ = view(u, model.idx_λ)
	b = view(u, model.idx_b)
	return @SVector [model.μ * λ[1] - sum(view(b, 1:2)),
					 model.μ * λ[2] - sum(view(b, 3:4))]
end

function maximum_dissipation(model::Biped{Discrete, FixedTime}, x⁺, u, h)
	q3 = view(x⁺, model.nq .+ (1:model.nq))
	q2 = view(x⁺, 1:model.nq)
	ψ = view(u, model.idx_ψ)
	ψ_stack = [ψ[1] * ones(2); ψ[2] * ones(2)]
	η = view(u, model.idx_η)
	return P_func(model, q3) * (q3 - q2) / h + ψ_stack - η
end

function maximum_dissipation(model::Biped{Discrete, FreeTime}, x⁺, u, h)
	q3 = view(x⁺, model.nq .+ (1:model.nq))
	q2 = view(x⁺, 1:model.nq)
	ψ = view(u, model.idx_ψ)
	ψ_stack = [ψ[1] * ones(2); ψ[2] * ones(2)]
	η = view(u, model.idx_η)
	h = u[end]
	return P_func(model, q3) * (q3 - q2) / h + ψ_stack - η
end

function no_slip(model::Biped{Discrete, FixedTime}, x⁺, u, h)
	q3 = view(x⁺, model.nq .+ (1:model.nq))
	q2 = view(x⁺, 1:model.nq)
	λ = view(u, model.idx_λ)
	s = view(u, model.idx_s)
	return s[1] - (λ' * _P_func(model, q3) * (q3 - q2) / h)[1]
end

function no_slip(model::Biped{Discrete, FreeTime}, x⁺, u, h)
	q3 = view(x⁺, model.nq .+ (1:model.nq))
	q2 = view(x⁺, 1:model.nq)
	λ = view(u, model.idx_λ)
	s = view(u, model.idx_s)
	h = u[end]
	return s[1] - (λ' * _P_func(model, q3) * (q3 - q2) / h)[1]
end

function fd(model::Biped{Discrete, FixedTime}, x⁺, x, u, w, h, t)
	q3 = view(x⁺, model.nq .+ (1:model.nq))
	q2⁺ = view(x⁺, 1:model.nq)
	q2⁻ = view(x, model.nq .+ (1:model.nq))
	q1 = view(x, 1:model.nq)
	u_ctrl = view(u, model.idx_u)
	λ = view(u, model.idx_λ)
	b = view(u, model.idx_b)

    SVector{14}([q2⁺ - q2⁻;
    ((1.0 / h) * (M_func(model, q1) * (SVector{7}(q2⁺) - SVector{7}(q1))
    - M_func(model, q2⁺) * (SVector{7}(q3) - SVector{7}(q2⁺)))
    + h * (transpose(B_func(model, q3)) * SVector{4}(u_ctrl)
    + transpose(N_func(model, q3)) * SVector{2}(λ)
    + transpose(P_func(model, q3)) * SVector{4}(b))
    - h * C_func(model, q3, (q3 - q2⁺) / h)
	+ h * w)])
end

function fd(model::Biped{Discrete, FreeTime}, x⁺, x, u, w, h, t)
	q3 = view(x⁺, model.nq .+ (1:model.nq))
	q2⁺ = view(x⁺, 1:model.nq)
	q2⁻ = view(x, model.nq .+ (1:model.nq))
	q1 = view(x, 1:model.nq)
	u_ctrl = view(u, model.idx_u)
	λ = view(u, model.idx_λ)
	b = view(u, model.idx_b)
	h = u[end]

    SVector{14}([q2⁺ - q2⁻;
    ((1.0 / h) * (M_func(model, q1) * (SVector{7}(q2⁺) - SVector{7}(q1))
    - M_func(model, q2⁺) * (SVector{7}(q3) - SVector{7}(q2⁺)))
    + h * (transpose(B_func(model, q3)) * SVector{4}(u_ctrl)
    + transpose(N_func(model, q3)) * SVector{2}(λ)
    + transpose(P_func(model, q3)) * SVector{4}(b))
    - h * C_func(model, q3, (q3 - q2⁺) / h)
	+ h * w)])
end

model = Biped{Discrete, FixedTime}(n, m, d,
			  g, μ,
			  l_torso, d_torso, m_torso, J_torso,
			  l_thigh, d_thigh, m_thigh, J_thigh,
			  l_calf, d_calf, m_calf, J_calf,
			  l_thigh, d_thigh, m_thigh, J_thigh,
			  l_calf, d_calf, m_calf, J_calf,
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

#NOTE: if a new model is instantiated, re-run the lines below
@variables z_sym[1:model.n]
l(z) = lagrangian(model, view(z, 1:model.nq), view(z, model.nq .+ (1:model.nq)))
_l = simplify.(l(z_sym))
_dL = ModelingToolkit.gradient(_l, z_sym)
_dLq = view(_dL, 1:model.nq)
_dLq̇ = view(_dL, model.nq .+ (1:model.nq))
_ddL = ModelingToolkit.sparsehessian(_l, z_sym)
_ddLq̇q = view(_ddL, model.nq .+ (1:model.nq), 1:model.nq)

dL = eval(ModelingToolkit.build_function(_dL, z_sym)[1])
dLq = eval(ModelingToolkit.build_function(_dLq, z_sym)[1])
dLq̇ = eval(ModelingToolkit.build_function(_dLq̇, z_sym)[1])
ddLq̇q = eval(ModelingToolkit.build_function(_ddLq̇q, z_sym)[1])
ddL = eval(ModelingToolkit.build_function(_ddL, z_sym)[1])

function C_func(model::Biped, q, q̇)
	ddLq̇q([q; q̇]) * q̇ - dLq([q; q̇])
end

qq = rand(model.nq)
vv = rand(model.nq)
norm(C_func(model, qq, vv) - _C_func(model, qq, vv))
dL(qq)

# visualization
function visualize!(vis, model::Biped, q;
      r = 0.035, Δt = 0.1)

	default_background!(vis)

	torso = Cylinder(Point3f0(0.0, 0.0, 0.0), Point3f0(0.0, 0.0, model.l_torso),
		convert(Float32, 0.025))
	setobject!(vis["torso"], torso,
		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

	thigh_1 = Cylinder(Point3f0(0.0,0.0,0.0), Point3f0(0.0, 0.0, model.l_thigh1),
		convert(Float32, 0.025))
	setobject!(vis["thigh1"], thigh_1,
		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

	calf_1 = Cylinder(Point3f0(0.0,0.0,0.0), Point3f0(0.0, 0.0, model.l_calf1),
		convert(Float32, 0.025))
	setobject!(vis["calf1"], calf_1,
		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

	thigh_2 = Cylinder(Point3f0(0.0,0.0,0.0), Point3f0(0.0, 0.0, model.l_thigh2),
		convert(Float32, 0.025))
	setobject!(vis["thigh2"], thigh_2,
		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

	calf_2 = Cylinder(Point3f0(0.0,0.0,0.0), Point3f0(0.0, 0.0, model.l_calf2),
		convert(Float32, 0.025))
	setobject!(vis["calf2"], calf_2,
		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

	setobject!(vis["foot1"], Sphere(Point3f0(0.0),
		convert(Float32, r)),
		MeshPhongMaterial(color = RGBA(1.0, 165.0 / 255.0, 0.0, 1.0)))
	setobject!(vis["foot2"], Sphere(Point3f0(0.0),
		convert(Float32, r)),
		MeshPhongMaterial(color = RGBA(1.0, 165.0 / 255.0, 0.0, 1.0)))
	setobject!(vis["knee1"], Sphere(Point3f0(0.0),
		convert(Float32, 0.025)),
		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))
	setobject!(vis["knee2"], Sphere(Point3f0(0.0),
		convert(Float32, 0.025)),
		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))
	setobject!(vis["hip"], Sphere(Point3f0(0.0),
		convert(Float32, 0.025)),
		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))
	setobject!(vis["torso_top"], Sphere(Point3f0(0.0),
		convert(Float32, 0.025)),
		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

	anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

	T = length(q)
	p_shift = [0.0; 0.0; r]
	for t = 1:T
		MeshCat.atframe(anim, t) do
			p = [q[t][1]; 0.0; q[t][2]] + p_shift

			k_torso = kinematics_1(model, q[t], body = :torso, mode = :ee)
			p_torso = [k_torso[1], 0.0, k_torso[2]] + p_shift

			k_thigh_1 = kinematics_1(model, q[t], body = :thigh_1, mode = :ee)
			p_thigh_1 = [k_thigh_1[1], 0.0, k_thigh_1[2]] + p_shift

			k_calf_1 = kinematics_2(model, q[t], body = :calf_1, mode = :ee)
			p_calf_1 = [k_calf_1[1], 0.0, k_calf_1[2]] + p_shift

			k_thigh_2 = kinematics_1(model, q[t], body = :thigh_2, mode = :ee)
			p_thigh_2 = [k_thigh_2[1], 0.0, k_thigh_2[2]] + p_shift

			k_calf_2 = kinematics_2(model, q[t], body = :calf_2, mode = :ee)
			p_calf_2 = [k_calf_2[1], 0.0, k_calf_2[2]] + p_shift

			settransform!(vis["thigh1"], cable_transform(p, p_thigh_1))
			settransform!(vis["calf1"], cable_transform(p_thigh_1, p_calf_1))
			settransform!(vis["thigh2"], cable_transform(p, p_thigh_2))
			settransform!(vis["calf2"], cable_transform(p_thigh_2, p_calf_2))
			settransform!(vis["torso"], cable_transform(p_torso,p))
			settransform!(vis["foot1"], Translation(p_calf_1))
			settransform!(vis["foot2"], Translation(p_calf_2))
			settransform!(vis["knee1"], Translation(p_thigh_1))
			settransform!(vis["knee2"], Translation(p_thigh_2))
			settransform!(vis["hip"], Translation(p))
			settransform!(vis["torso_top"], Translation(p_torso))


		end
	end

	settransform!(vis["/Cameras/default"],
	   compose(Translation(0.0 , 1.0 , -1.0), LinearMap(RotZ(-pi / 2.0))))

	MeshCat.setanimation!(vis, anim)
end

function initial_configuration(model, θ_torso, θ_thigh_1, θ_leg_1, θ_thigh_2)
    q1 = zeros(model.nq)
    q1[3] = θ_torso
    q1[4] = θ_thigh_1
    q1[5] = θ_leg_1
    z1 = model.l_thigh1 * cos(q1[4]) + model.l_calf1 * cos(q1[5])
    q1[6] = θ_thigh_2
    q1[7] = -1.0 * acos((z1 - model.l_thigh2 * cos(q1[6])) / model.l_calf2)
    q1[2] = z1

    p1 = kinematics_2(model, q1, body = :calf_1, mode = :ee)
    p2 = kinematics_2(model, q1, body = :calf_2, mode = :ee)
    @show stride = abs(p1[1] - p2[1])

    q1[1] = -1.0 * p1[1]

    qM = copy(q1)
    qM[4] = q1[6]
    qM[5] = q1[7]
    qM[6] = q1[4]
    qM[7] = q1[5]
    qM[1] = abs(p2[1])

    pM_1 = kinematics_2(model, qM, body = :calf_1, mode = :ee)
    pM_2 = kinematics_2(model, qM, body = :calf_2, mode = :ee)

    qT = copy(q1)
    qT[1] = 2 * stride

    pT_1 = kinematics_2(model, qT, body = :calf_1, mode = :ee)
    pT_2 = kinematics_2(model, qT, body = :calf_2, mode = :ee)

    return q1, qM, qT
end
