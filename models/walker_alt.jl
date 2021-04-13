struct Walker{I, T} <: Model{I, T}
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

		# foot
	l_foot1 # toe length
	d_foot1 # heel length
	m_foot1
	J_foot1

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

		# foot
	l_foot2 # toe length
	d_foot2 # heel length
	m_foot2
	J_foot2

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

    joint_friction
end

# Dimensions
nq = 2 + 5            # configuration dimension
nu = 2                   # control dimension
nc = 4                    # number of contact points
nf = 2                    # number of parameters for friction cone
nb = nc * nf              # number of friction parameters
ns = 1                    # slack

# World parameters
μ_world = 0.1      # coefficient of friction
g_world = 9.81     # gravity
joint_friction = 0.1

# Model parameters
m_torso = 0.5 + 0.48 * 2.0
m_thigh = 0.8112
m_calf = 0.3037
m_foot = 0.4 * m_calf

l_torso = 0.15 + 0.15
l_thigh = 0.2755
l_calf = 0.308
l_foot = 0.4 * l_calf

d_torso = 0.0342
d_thigh = 0.2176
d_calf = 0.1445
d_foot = 0.4 * d_calf

J_torso = 0.0029
J_thigh = 0.00709
J_calf = 0.00398
J_foot = 0.4 * J_calf # 1.0 / 12.0 * m_foot * (l_foot + d_foot)^2.0

n = 2 * nq
m = nu + nc + nb + nc + nb + ns
d = nq

idx_u = (1:nu)
idx_λ = nu .+ (1:nc)
idx_b = nu + nc .+ (1:nb)
idx_ψ = nu + nc + nb .+ (1:nc)
idx_η = nu + nc + nb + nc .+ (1:nb)
idx_s = nu + nc + nb + nc + nb .+ (1:ns)

# configuration limits
qL = -Inf * ones(nq)
qU = Inf * ones(nq)

# control limits
uL = -100.0 * ones(nu)
uU = 100.0 * ones(nu)
function kinematics_1(model::Walker, q; body = :torso, mode = :ee)
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

function jacobian_1(model::Walker, q; body = :torso, mode = :ee)
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

function kinematics_2(model::Walker, q; body = :calf_1, mode = :ee)

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

function jacobian_2(model::Walker, q; body = :calf_1, mode = :ee)

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
#
# function kinematics_3(model::Walker, q; body = :foot_1, mode = :ee)
#
# 	if body == :foot_1
# 		p = kinematics_2(model, q, body = :calf_1, mode = :ee)
#
# 		θb = q[8]
#
# 		lb = model.l_foot1
# 		db = model.d_foot1
# 		cb = 0.5 * (model.l_foot1 - model.d_foot1)
# 	elseif body == :foot_2
# 		p = kinematics_2(model, q, body = :calf_2, mode = :ee)
#
# 		θb = q[9]
#
# 		lb = model.l_foot2
# 		db = model.d_foot2
# 		cb = 0.5 * (model.l_foot2 - model.d_foot2)
# 	else
# 		@error "incorrect body specification"
# 	end
#
# 	if mode == :toe
# 		return p + [lb * sin(θb); -1.0 * lb * cos(θb)]
# 	elseif mode == :heel
# 		return p + [-db * sin(θb); 1.0 * db * cos(θb)]
# 	elseif mode == :com
# 		return p + [cb * sin(θb); -1.0 * cb * cos(θb)]
# 	else
# 		@error "incorrect mode specification"
# 	end
# end
#
# function jacobian_3(model::Walker, q; body = :foot_1, mode = :ee)
#
# 	if body == :foot_1
# 		jac = jacobian_2(model, q, body = :calf_1, mode = :ee)
#
# 		θb = q[8]
#
# 		if mode == :toe
# 			r = model.l_foot1
# 		elseif mode == :heel
# 			r = -1.0 * model.d_foot1
# 		elseif mode == :com
# 			r = 0.5 * (model.l_foot1 - model.d_foot1)
# 		else
# 			@error "incorrect mode specification"
# 		end
#
# 		jac[1, 8] += r * cos(θb)
# 		jac[2, 8] += r * sin(θb)
#
# 	elseif body == :foot_2
# 		jac = jacobian_2(model, q, body = :calf_2, mode = :ee)
#
# 		θb = q[9]
#
# 		if mode == :toe
# 			r = model.l_foot2
# 		elseif mode == :heel
# 			r = -1.0 * model.d_foot2
# 		elseif mode == :com
# 			r = 0.5 * (model.l_foot2 - model.d_foot2)
# 		else
# 			@error "incorrect mode specification"
# 		end
# 		jac[1, 9] += r * cos(θb)
# 		jac[2, 9] += r * sin(θb)
#
# 	else
# 		@error "incorrect body specification"
# 	end
#
# 	return jac
# end

# Lagrangian

function lagrangian(model::Walker, q, q̇)
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

	# # foot 1
	# p_foot_1 = kinematics_3(model, q, body = :foot_1, mode = :com)
	# J_foot_1 = jacobian_3(model, q, body = :foot_1, mode = :com)
	# v_foot_1 = J_foot_1 * q̇
	#
	# L += 0.5 * model.m_foot1 * transpose(v_foot_1) * v_foot_1
	# L += 0.5 * model.J_foot1 * q̇[8]^2.0
	# L -= model.m_foot1 * model.g * p_foot_1[2]

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

	# # foot 2
	# p_foot_2 = kinematics_3(model, q, body = :foot_2, mode = :com)
	# J_foot_2 = jacobian_3(model, q, body = :foot_2, mode = :com)
	# v_foot_2 = J_foot_2 * q̇
	#
	# L += 0.5 * model.m_foot2 * transpose(v_foot_2) * v_foot_2
	# L += 0.5 * model.J_foot2 * q̇[9]^2.0
	# L -= model.m_foot2 * model.g * p_foot_2[2]

	return L
end

function _dLdq(model::Walker, q, q̇)
	Lq(x) = lagrangian(model, x, q̇)
	ForwardDiff.gradient(Lq, q)
end

function _dLdq̇(model::Walker, q, q̇)
	Lq̇(x) = lagrangian(model, q, x)
	ForwardDiff.gradient(Lq̇, q̇)
end

function _C_func(model::Walker, q, q̇)
	tmp_q(z) = _dLdq̇(model, z, q̇)
	tmp_q̇(z) = _dLdq̇(model, q, z)

	ForwardDiff.jacobian(tmp_q, q) * q̇ - _dLdq(model, q, q̇)
end

# Methods
function M_func(model::Walker, q)
	M = Diagonal([0.0, 0.0,
		model.J_torso,
		model.J_thigh1, model.J_calf1,
		model.J_thigh2, model.J_calf2])#,
		# model.J_foot1, model.J_foot2])

	# torso
	J_torso = jacobian_1(model, q, body = :torso, mode = :com)
	M += model.m_torso * transpose(J_torso) * J_torso

	# thigh 1
	J_thigh_1 = jacobian_1(model, q, body = :thigh_1, mode = :com)
	M += model.m_thigh1 * transpose(J_thigh_1) * J_thigh_1

	# leg 1
	J_calf_1 = jacobian_2(model, q, body = :calf_1, mode = :com)
	M += model.m_calf1 * transpose(J_calf_1) * J_calf_1

	# # foot 1
	# J_foot_1 = jacobian_3(model, q, body = :foot_1, mode = :com)
	# M += model.m_foot1 * transpose(J_foot_1) * J_foot_1

	# thigh 2
	J_thigh_2 = jacobian_1(model, q, body = :thigh_2, mode = :com)
	M += model.m_thigh2 * transpose(J_thigh_2) * J_thigh_2

	# leg 2
	J_calf_2 = jacobian_2(model, q, body = :calf_2, mode = :com)
	M += model.m_calf2 * transpose(J_calf_2) * J_calf_2

	# # foot 2
	# J_foot_2 = jacobian_3(model, q, body = :foot_2, mode = :com)
	# M += model.m_foot2 * transpose(J_foot_2) * J_foot_2

	return M
end

function ϕ_func(model::Walker, q)
	p_toe_1 = kinematics_3(model, q, body = :foot_1, mode = :toe)
	p_heel_1 = kinematics_3(model, q, body = :foot_1, mode = :heel)
	p_toe_2 = kinematics_3(model, q, body = :foot_2, mode = :toe)
	p_heel_2 = kinematics_3(model, q, body = :foot_2, mode = :heel)

	@SVector [p_toe_1[2], p_heel_1[2], p_toe_2[2], p_heel_2[2]]
end

function B_func(model::Walker, q)
	@SMatrix [0.0 0.0 1.0 0.0 0.0 0.0 0.0;
			  0.0 0.0 0.0 1.0 0.0 0.0 0.0;
			  0.0 0.0 0.0 0.0 1.0 0.0 0.0;
			  0.0 0.0 0.0 0.0 0.0 1.0 0.0;
			  0.0 0.0 0.0 0.0 0.0 0.0 1.0]

end

function N_func(model::Walker, q)
	J_toe_1 = jacobian_3(model, q, body = :foot_1, mode = :toe)
	J_heel_1 = jacobian_3(model, q, body = :foot_1, mode = :heel)
	J_toe_2 = jacobian_3(model, q, body = :foot_2, mode = :toe)
	J_heel_2 = jacobian_3(model, q, body = :foot_2, mode = :heel)

	return [view(J_toe_1, 2:2, :);
			view(J_heel_1, 2:2, :);
			view(J_toe_2, 2:2, :);
			view(J_heel_2, 2:2, :)]
end

function _P_func(model::Walker, q)
	J_toe_1 = jacobian_3(model, q, body = :foot_1, mode = :toe)
	J_heel_1 = jacobian_3(model, q, body = :foot_1, mode = :heel)
	J_toe_2 = jacobian_3(model, q, body = :foot_2, mode = :toe)
	J_heel_2 = jacobian_3(model, q, body = :foot_2, mode = :heel)

	return [view(J_toe_1, 1:1, :);
			view(J_heel_1, 1:1, :);
			view(J_toe_2, 1:1, :);
			view(J_heel_2, 1:1, :)]
end

function P_func(model::Walker, q)
	J_toe_1 = jacobian_3(model, q, body = :foot_1, mode = :toe)
	J_heel_1 = jacobian_3(model, q, body = :foot_1, mode = :heel)
	J_toe_2 = jacobian_3(model, q, body = :foot_2, mode = :toe)
	J_heel_2 = jacobian_3(model, q, body = :foot_2, mode = :heel)
	map = [1.0; -1.0]

	return [map * view(J_toe_1, 1:1, :);
			map * view(J_heel_1, 1:1, :);
			map * view(J_toe_2, 1:1, :);
			map * view(J_heel_2, 1:1, :)]
end

function friction_cone(model::Walker, u)
	λ = view(u, model.idx_λ)
	b = view(u, model.idx_b)

	return @SVector [model.μ * λ[1] - sum(view(b, 1:2)),
					 model.μ * λ[2] - sum(view(b, 3:4)),
					 model.μ * λ[3] - sum(view(b, 5:6)),
					 model.μ * λ[4] - sum(view(b, 7:8))]
end

function maximum_dissipation(model::Walker{Discrete, FixedTime}, x⁺, u, h)
	q3 = view(x⁺, model.nq .+ (1:model.nq))
	q2 = view(x⁺, 1:model.nq)
	ψ = view(u, model.idx_ψ)
	ψ_stack = [ψ[1] * ones(2); ψ[2] * ones(2); ψ[3] * ones(2); ψ[4] * ones(2)]
	η = view(u, model.idx_η)
	return P_func(model, q3) * (q3 - q2) / h + ψ_stack - η
end

function maximum_dissipation(model::Walker{Discrete, FreeTime}, x⁺, u, h)
	q3 = view(x⁺, model.nq .+ (1:model.nq))
	q2 = view(x⁺, 1:model.nq)
	ψ = view(u, model.idx_ψ)
	ψ_stack = [ψ[1] * ones(2); ψ[2] * ones(2); ψ[3] * ones(2); ψ[4] * ones(2)]
	η = view(u, model.idx_η)
	h = u[end]
	return P_func(model, q3) * (q3 - q2) / h + ψ_stack - η
end

function no_slip(model::Walker{Discrete, FixedTime}, x⁺, u, h)
	q3 = view(x⁺, model.nq .+ (1:model.nq))
	q2 = view(x⁺, 1:model.nq)
	λ = view(u, model.idx_λ)
	s = view(u, model.idx_s)
	return s[1] - (λ' * _P_func(model, q3) * (q3 - q2) / h)[1]
end

function no_slip(model::Walker{Discrete, FreeTime}, x⁺, u, h)
	q3 = view(x⁺, model.nq .+ (1:model.nq))
	q2 = view(x⁺, 1:model.nq)
	λ = view(u, model.idx_λ)
	s = view(u, model.idx_s)
	h = u[end]
	return s[1] - (λ' * _P_func(model, q3) * (q3 - q2) / h)[1]
end

function lagrangian_derivatives(model, q, v)
	D1L = -1.0 * C_func(model, q, v)
    D2L = M_func(model, q) * v
	return D1L, D2L
end

function fd(model::Walker{Discrete, FixedTime}, x⁺, x, u, w, h, t)
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

	joint_friction = model.joint_friction * vm2
	joint_friction[1:3] .= 0.0

	D1L1, D2L1 = lagrangian_derivatives(model, qm1, vm1)
	D1L2, D2L2 = lagrangian_derivatives(model, qm2, vm2)

	[q2⁺ - q2⁻;
     (0.5 * h * D1L1 + D2L1 + 0.5 * h * D1L2 - D2L2
     + transpose(B_func(model, qm2)) * SVector{7}(u_ctrl)
     + transpose(N_func(model, q3)) * SVector{4}(λ)
     + transpose(P_func(model, q3)) * SVector{8}(b)
     - h * joint_friction)]
end

function fd(model::Walker{Discrete, FreeTime}, x⁺, x, u, w, h, t)
	q3 = view(x⁺, model.nq .+ (1:model.nq))
	q2⁺ = view(x⁺, 1:model.nq)
	q2⁻ = view(x, model.nq .+ (1:model.nq))
	q1 = view(x, 1:model.nq)
	u_ctrl = view(u, model.idx_u)
	λ = view(u, model.idx_λ)
	b = view(u, model.idx_b)
	h = u[end]

	qm1 = 0.5 * (q1 + q2⁺)
    vm1 = (q2⁺ - q1) / h
    qm2 = 0.5 * (q2⁺ + q3)
    vm2 = (q3 - q2⁺) / h

	joint_friction = model.joint_friction * vm2
	joint_friction[1:3] .= 0.0

	D1L1, D2L1 = lagrangian_derivatives(model, qm1, vm1)
	D1L2, D2L2 = lagrangian_derivatives(model, qm2, vm2)

	[q2⁺ - q2⁻;
     (0.5 * h * D1L1 + D2L1 + 0.5 * h * D1L2 - D2L2
     + transpose(B_func(model, qm2)) * SVector{7}(u_ctrl)
     + transpose(N_func(model, q3)) * SVector{4}(λ)
     + transpose(P_func(model, q3)) * SVector{8}(b)
     - h * joint_friction)]
end

model = Walker{Discrete, FixedTime}(n, m, d,
			  g_world, μ_world,
			  l_torso, d_torso, m_torso, J_torso,
			  l_thigh, d_thigh, m_thigh, J_thigh,
			  l_calf, d_calf, m_calf, J_calf,
			  l_foot, d_foot, m_foot, J_foot,
			  l_thigh, d_thigh, m_thigh, J_thigh,
			  l_calf, d_calf, m_calf, J_calf,
			  l_foot, d_foot, m_foot, J_foot,
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
			  idx_s,
			  0.0 * joint_friction)

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

function C_func(model::Walker, q, q̇)
	ddLq̇q([q; q̇]) * q̇ - dLq([q; q̇])
end

norm(M_func(model, ones(nq)))
C_func(model, ones(nq), ones(nq))


# qq = rand(model.nq)
# vv = rand(model.nq)
# norm(C_func(model, qq, vv) - _C_func(model, qq, vv))
# dL(qq)

# visualization
function visualize!(vis, model::Walker, q;
      r = 0.040, Δt = 0.1)

	default_background!(vis)

	torso = Cylinder(Point3f0(0.0, 0.0, 0.0), Point3f0(0.0, 0.0, model.l_torso),
		convert(Float32, 0.035))
	setobject!(vis["torso"], torso,
		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

	thigh_1 = Cylinder(Point3f0(0.0,0.0,0.0), Point3f0(0.0, 0.0, model.l_thigh1),
		convert(Float32, 0.035))
	setobject!(vis["thigh1"], thigh_1,
		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

	calf_1 = Cylinder(Point3f0(0.0,0.0,0.0), Point3f0(0.0, 0.0, model.l_calf1),
		convert(Float32, 0.035))
	setobject!(vis["calf1"], calf_1,
		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

	foot_1 = Cylinder(Point3f0(0.0,0.0,0.0),
		Point3f0(0.0, 0.0, model.l_foot1 + model.d_foot1),
		convert(Float32, 0.035))
	setobject!(vis["foot1"], foot_1,
		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

	thigh_2 = Cylinder(Point3f0(0.0,0.0,0.0), Point3f0(0.0, 0.0, model.l_thigh2),
		convert(Float32, 0.035))
	setobject!(vis["thigh2"], thigh_2,
		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

	calf_2 = Cylinder(Point3f0(0.0,0.0,0.0), Point3f0(0.0, 0.0, model.l_calf2),
		convert(Float32, 0.035))
	setobject!(vis["calf2"], calf_2,
		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

	foot_2 = Cylinder(Point3f0(0.0,0.0,0.0),
		Point3f0(0.0, 0.0, model.l_foot2 + model.d_foot2),
		convert(Float32, 0.035))
	setobject!(vis["foot2"], foot_2,
		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

	setobject!(vis["heel1"], Sphere(Point3f0(0.0),
		convert(Float32, r)),
		MeshPhongMaterial(color = RGBA(1.0, 165.0 / 255.0, 0.0, 1.0)))
	setobject!(vis["heel2"], Sphere(Point3f0(0.0),
		convert(Float32, r)),
		MeshPhongMaterial(color = RGBA(1.0, 165.0 / 255.0, 0.0, 1.0)))
	setobject!(vis["toe1"], Sphere(Point3f0(0.0),
		convert(Float32, r)),
		MeshPhongMaterial(color = RGBA(1.0, 165.0 / 255.0, 0.0, 1.0)))
	setobject!(vis["toe2"], Sphere(Point3f0(0.0),
		convert(Float32, r)),
		MeshPhongMaterial(color = RGBA(1.0, 165.0 / 255.0, 0.0, 1.0)))
	setobject!(vis["knee1"], Sphere(Point3f0(0.0),
		convert(Float32, 0.035)),
		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))
	setobject!(vis["knee2"], Sphere(Point3f0(0.0),
		convert(Float32, 0.035)),
		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))
	setobject!(vis["hip"], Sphere(Point3f0(0.0),
		convert(Float32, 0.035)),
		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))
	setobject!(vis["torso_top"], Sphere(Point3f0(0.0),
		convert(Float32, 0.035)),
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

			k_toe_1 = kinematics_3(model, q[t], body = :foot_1, mode = :toe)
			p_toe_1 = [k_toe_1[1], 0.0, k_toe_1[2]] + p_shift

			k_heel_1 = kinematics_3(model, q[t], body = :foot_1, mode = :heel)
			p_heel_1 = [k_heel_1[1], 0.0, k_heel_1[2]] + p_shift

			k_toe_2 = kinematics_3(model, q[t], body = :foot_2, mode = :toe)
			p_toe_2 = [k_toe_2[1], 0.0, k_toe_2[2]] + p_shift

			k_heel_2 = kinematics_3(model, q[t], body = :foot_2, mode = :heel)
			p_heel_2 = [k_heel_2[1], 0.0, k_heel_2[2]] + p_shift

			settransform!(vis["thigh1"], cable_transform(p, p_thigh_1))
			settransform!(vis["calf1"], cable_transform(p_thigh_1, p_calf_1))
			settransform!(vis["foot1"], cable_transform(p_toe_1, p_heel_1))

			settransform!(vis["thigh2"], cable_transform(p, p_thigh_2))
			settransform!(vis["calf2"], cable_transform(p_thigh_2, p_calf_2))
			settransform!(vis["foot2"], cable_transform(p_toe_2, p_heel_2))

			settransform!(vis["torso"], cable_transform(p_torso,p))
			settransform!(vis["heel1"], Translation(p_heel_1))
			settransform!(vis["heel2"], Translation(p_heel_2))
			settransform!(vis["toe1"], Translation(p_toe_1))
			settransform!(vis["toe2"], Translation(p_toe_2))
			settransform!(vis["knee1"], Translation(p_thigh_1))
			settransform!(vis["knee2"], Translation(p_thigh_2))
			settransform!(vis["hip"], Translation(p))
			settransform!(vis["torso_top"], Translation(p_torso))
		end
	end

	settransform!(vis["/Cameras/default"],
	    compose(Translation(0.0, 0.0, -1.0), LinearMap(RotZ(-pi / 2.0))))

	MeshCat.setanimation!(vis, anim)
end
