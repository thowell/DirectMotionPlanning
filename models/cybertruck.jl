struct CYBERTRUCK{I, T} <: Model{I, T}
    n::Int
    m::Int
    d::Int

    mass
    J
    μ
    g

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
nq = 4 # configuration dim
nu = 2 # control dim
nc = 1 # number of contact points
nf = 4 # number of faces for friction cone pyramid
nb = nc * nf
ns = 1

# Parameters
g = 9.81   # gravity
μ = 0.5    # coefficient of friction
mass = 1.0 # body mass
J = 1.0    # body inertia

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
M_func(model::CYBERTRUCK, q) = Diagonal(@SVector [model.mass, model.mass, model.mass, model.J])
G_func(model::CYBERTRUCK, q) = @SVector [0.0, 0.0, model.mass * model.g, 0.0]

function ϕ_func(::CYBERTRUCK, q)
    return @SVector [q[3]]
end

N_func(::CYBERTRUCK, q) = @SMatrix [0.0 0.0 1.0 0.0]

P_func(::CYBERTRUCK, q) = @SMatrix [1.0 0.0 0.0 0.0;
                                     -1.0 0.0 0.0 0.0;
                                     0.0 1.0 0.0 0.0;
                                     0.0 -1.0 0.0 0.0]

B_func(::CYBERTRUCK, q) = @SMatrix [cos(q[4]) sin(q[4]) 0.0 0.0;
                                    0.0 0.0 0.0 1.0]


function friction_cone(model::CYBERTRUCK, u)
	λ = u[model.idx_λ]
	b = u[model.idx_b]
	return @SVector [model.μ * λ[1] - sum(b)]
end

function maximum_dissipation(model::CYBERTRUCK, x⁺, u, h)
	q3 = x⁺[model.nq .+ (1:model.nq)]
	q2 = x⁺[1:model.nq]
	ψ = u[model.idx_ψ]
	ψ_stack = ψ[1] * ones(4)
	η = u[model.idx_η]
	return P_func(model, q3) * (q3 - q2) / h + ψ_stack - η
end

function fd(model::CYBERTRUCK{Discrete, FixedTime}, x⁺, x, u, w, h, t)
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
    + transpose(B_func(model, q3)) * SVector{2}(u_ctrl)
    + transpose(N_func(model, q3)) * SVector{1}(λ)
    + transpose(P_func(model, q3)) * SVector{4}(b)
    - h * G_func(model, q2⁺))]
end

model = CYBERTRUCK{Discrete, FixedTime}(n, m, d,
				   mass, J, μ, g,
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

function visualize!(vis, model::CYBERTRUCK, q;
		Δt = 0.1, scenario = :pp)

	default_background!(vis)

    obj_path = joinpath(pwd(), "models/cybertruck/cybertruck.obj")
    mtl_path = joinpath(pwd(), "models/cybertruck/cybertruck.mtl")

    ctm = ModifiedMeshFileObject(obj_path, mtl_path, scale=0.1)
    setobject!(vis["cybertruck"], ctm)
    settransform!(vis["cybertruck"], LinearMap(RotZ(pi) * RotX(pi / 2.0)))

    anim = MeshCat.Animation(convert(Int,floor(1.0 / Δt)))

    for t = 1:length(q)
        MeshCat.atframe(anim, t) do
            settransform!(vis["cybertruck"], compose(Translation(q[t][1:3]...),
				LinearMap(RotZ(q[t][4] + pi) * RotY(0.0) * RotX(pi / 2.0))))
        end
    end

	if scenario == :pp
		settransform!(vis["/Cameras/default"],
			compose(Translation(2.0, 0.0, 1.0),LinearMap(RotZ(0.0))))
		# add parked cars
		obj_path = joinpath(pwd(),"models/cybertruck/cybertruck.obj")
		mtl_path = joinpath(pwd(),"models/cybertruck/cybertruck.mtl")

		ctm = ModifiedMeshFileObject(obj_path, mtl_path, scale = 0.1)

		setobject!(vis["cybertruck_park1"], ctm)
		settransform!(vis["cybertruck_park1"],
		    compose(Translation([p_car1[1]; p_car1[2]; 0.0]),
		    LinearMap(RotZ(pi + pi / 2) * RotX(pi / 2.0))))

		setobject!(vis["cybertruck_park2"], ctm)
		settransform!(vis["cybertruck_park2"],
		    compose(Translation([3.0; -0.65; 0.0]),
		    LinearMap(RotZ(pi + pi / 2) * RotX(pi / 2.0))))
	end

    MeshCat.setanimation!(vis, anim)
end
