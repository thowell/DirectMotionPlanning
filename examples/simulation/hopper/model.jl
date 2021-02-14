"""
    hopper dynamics
    - 3D particle subject to contact forces

    - configuration: q = (x, y, z, tx, ty, tz, r) ∈ R⁶
    - orientation: modified Rodrigues angles
    - impact force (magnitude): n ∈ R₊
    - friction force: b ∈ R²
        - contact force: λ = (b, n) ∈ R⁴₊ × R₊
        - friction coefficient: μ ∈ R₊
    Discrete Mechanics and Variational Integrators
        pg. 363
"""
struct Hopper3D
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
end

# Kinematics
function kinematics(::Hopper3D, q)
	p = view(q, 1:3)
	R = MRP(view(q, 4:6)...)
	p + R*[0.0; 0.0; -1.0 * q[7]]
end

# mass matrix
function mass_matrix(model)
	mb = model.mb
	ml = model.ml
	Jb = model.Jb
	Jl = model.Jl

	Diagonal(@SVector [mb + ml, mb + ml, mb + ml,
					   Jb + Jl, Jb + Jl, Jb + Jl,
					   ml])
end

# gravity
function gravity(model)
	@SVector [0.0, 0.0, (model.mb + model.ml) * model.g, 0.0, 0.0, 0.0, 0.0]
end

# signed distance function
function signed_distance(model, q)
    kinematics(model, q)[3]
end

# contact force Jacobian
function jacobian(model, q)
	k(z) = kinematics(model, z)
    ForwardDiff.jacobian(k, q)
end

function input(::Hopper3D, q)
    rot = view(q, 4:6)
    R = MRP(rot...)
    @SMatrix [0.0 0.0 0.0 R[1,1] R[2,1] R[3,1] 0.0;
              0.0 0.0 0.0 R[1,2] R[2,2] R[3,2] 0.0;
			  R[1,3] R[2,3] R[3,3] 0.0 0.0 0.0 1.0]
end

# dynamics
function dynamics(model, q1, q2, q3, u, λ, h)
    nq = model.nq
    SVector{nq}(mass_matrix(model) * (2.0 * q2 - q1 - q3) / h
        - h * gravity(model)
        + h * jacobian(model, q3)' * λ
		+ h * input(model, q3)' * u)
end

# Model parameters
r = 0.5
qL = -Inf * ones(7)
qU = Inf * ones(7)
qL[3] = 0.0
qL[7] = 0.1
qU[7] = r

g = 9.81 # gravity
μ = 1.0  # coefficient of friction
mb = 1.0 # body mass
ml = 0.1  # leg mass
Jb = 0.25 # body inertia
Jl = 0.025 # leg inertia

model = Hopper3D(14, 3, 0,
			mb, ml, Jb, Jl,
			μ, g,
			qL, qU,
            7, 3)
