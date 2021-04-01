"""
	Particle
"""
struct Particle{I, T} <: Model{I, T}
	n::Int
	m::Int
	d::Int

    mass
	g
    μ

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
nq = 3                    # configuration dimension
nu = 3                    # control dimension
nc = 1                    # number of contact points
nf = 4                    # number of parameters for friction cone
nb = nc * nf
ns = 1

# Parameters
μ = 0.5      # coefficient of friction
mass = 1.0   # mass
g = 9.81

n = 2 * nq
m = nu + nc + nb + nc + nb + ns
d = 0

idx_u = (1:nu)
idx_λ = nu .+ (1:nc)
idx_b = nu + nc .+ (1:nb)
idx_ψ = nu + nc + nb .+ (1:nc)
idx_η = nu + nc + nb + nc .+ (1:nb)
idx_s = nu + nc + nb + nc + nb .+ (1:ns)

# Methods
function M_func(model::Particle, q)
    @SMatrix [model.mass 0.0 0.0;
              0.0 model.mass 0.0;
              0.0 0.0 model.mass]
end

C_func(model::Particle, q, q̇) = @SVector [0.0, 0.0, model.mass * model.g]

function ϕ_func(::Particle, q)
    @SVector[q[3]]
end

B_func(::Particle, q) = @SMatrix [1.0 0.0 0.0;
                                  0.0 1.0 0.0;
                                  0.0 0.0 1.0]

N_func(::Particle, q) = @SMatrix [0.0 0.0 1.0]

function _P_func(model::Particle, q)
   return @SMatrix [1.0 0.0 0.0;
                    0.0 1.0 0.0]
end

function P_func(model::Particle, q)
   return @SMatrix [1.0 0.0 0.0;
                    0.0 1.0 0.0;
                    -1.0 0.0 0.0;
                    0.0 -1.0 0.0]
end

function lagrangian_derivatives(model, q, v)
	D1L = -1.0 * C_func(model, q, v)
    D2L = M_func(model, q) * v
	return D1L, D2L
end


function friction_cone(model::Particle, u)
	λ = u[model.idx_λ]
	b = u[model.idx_b]
	return @SVector [model.μ * λ[1] - sum(b)]
end

function maximum_dissipation(model::Particle, x⁺, u, h)
	q3 = x⁺[model.nq .+ (1:model.nq)]
	q2 = x⁺[1:model.nq]
	ψ = u[model.idx_ψ]
	ψ_stack = ψ[1] * ones(4)
	η = u[model.idx_η]
	return P_func(model, q3) * (q3 - q2) / h + ψ_stack - η
end

function no_slip(model::Particle, x⁺, u, h)
	q3 = view(x⁺, model.nq .+ (1:model.nq))
	q2 = view(x⁺, 1:model.nq)
	λ = view(u, model.idx_λ)
	s = view(u, model.idx_s)
	λ_stack = λ[1] * ones(2)
	return (λ_stack' * _P_func(model, q3) * (q3 - q2) / h)[1]
end

function fd(model::Particle{Discrete, FixedTime}, x⁺, x, u, w, h, t)
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
	   + transpose(B_func(model, qm2)) * u_ctrl
	   + transpose(N_func(model, q3)) * λ
	   + transpose(P_func(model, q3)) * b)]
end

model = Particle{Discrete, FixedTime}(n, m, d,
				 mass, g, μ,
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

function visualize!(vis, model::Particle, q;
	Δt = 0.1, r = 0.25)

	default_background!(vis)
    setobject!(vis["particle"],
		Rect(Vec(0, 0, 0),Vec(2r, 2r, 2r)),
		MeshPhongMaterial(color = RGBA(1.0, 165.0 / 255.0, 0, 1.0)))

    anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

    for t = 1:length(q)
        MeshCat.atframe(anim, t) do
            settransform!(vis["particle"], Translation(q[t][1:3]...))
        end
    end

	settransform!(vis["/Cameras/default"],
	    compose(Translation(-2.5, 7.5, 1.0),LinearMap(RotZ(0.0))))

    MeshCat.setanimation!(vis, anim)
end
