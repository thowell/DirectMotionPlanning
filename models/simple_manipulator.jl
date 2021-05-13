# 2-link revolute-revolute manipulator + particle
mutable struct RRParticle{I, T} <: Model{I, T}
    n::Int
    m::Int
    d::Int

    m1
    J1
    l1

    m2
    J2
    l2

    mp

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

function kinematics_mid(model::RRParticle, q)
    @SVector [model.l1 * cos(q[1]),
              model.l1 * sin(q[1])]
end

function kinematics_ee(model::RRParticle, q)
    @SVector [model.l1 * cos(q[1]) + model.l2 * cos(q[1] + q[2]),
              model.l1 * sin(q[1]) + model.l2 * sin(q[1] + q[2])]
end

# Dimensions
nq = 4 # configuration dim
nu = 2
nc = 2 # number of contact points
nf = 2 # number of faces for friction cone pyramid
nb = nc * nf
ns = 1

# Parameters
μ = 0.5  # coefficient of friction
m1 = 1.0 # mass link 1
J1 = 1.0 # inertia link 1
l1 = 1.0 # length link 1
m2 = 1.0 # mass link 2
J2 = 1.0 # inertia link 2
l2 = 1.0 # length link 2
mp = 1.0 # mass particle

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
function M_func(model::RRParticle, q)
    a = (model.m1 * model.l1 * model.l1
         + model.J1
         + model.m2 * (model.l1 * model.l1 + model.l2 * model.l2 + 2.0 * model.l1 * model.l2 * cos(q[2]))
         + model.J2)
    b = model.m2 * (model.l2 * model.l2 + model.l1 * model.l2 * cos(q[2])) + model.J2
    c = model.l2 * model.l2 * model.m2 + model.J2

    @SMatrix [a b 0.0 0.0;
              b c 0.0 0.0;
              0.0 0.0 model.mp 0.0;
              0.0 0.0 0.0 model.mp]
end

function G_func(model::RRParticle, q)
    @SVector [((model.m1 + model.m2) * model.l1 * cos(q[1])
                + model.m2 * model.l2 * cos(q[1] + q[2])) * model.g,
               model.m2 * model.l2 * cos(q[1] + q[2]) * model.g,
               0.0,
               model.mp * model.g
             ]
end

function C_func(model::RRParticle, q⁻, q⁺, h)
    v = (q⁺ - q⁻) / h

    @SVector [(-2.0 * model.l1 * model.l2 * model.m2 * sin(q⁻[2]) * v[1] * v[2]
               + -1.0 * model.l1 * model.l2 * model.m2 * sin(q⁻[2]) * v[2] * v[2]),
               model.l1 * model.l2 * model.m2 * sin(q⁻[2]) * v[1] * v[1],
               0.0,
               0.0
              ]
end

function ϕ_func(model::RRParticle, q)
    x = q[3]
    z = q[4]
    px = model.l1 * cos(q[1]) + model.l2 * cos(q[1] + q[2])
    pz = model.l1 * sin(q[1]) + model.l2 * sin(q[1] + q[2])

    @SVector [(px - x) * (px - x) + (pz - z) * (pz - z), z]
end

B_func(::RRParticle, q) = @SMatrix [1.0 0.0 0.0 0.0;
                              0.0 1.0 0.0 0.0]

function N_func(model::RRParticle, q)
    x = q[3]
    z = q[4]
    px = model.l1 * cos(q[1]) + model.l2 * cos(q[1] + q[2])
    pz = model.l1 * sin(q[1]) + model.l2 * sin(q[1] + q[2])

    pxq1 = -1.0 * model.l1 * sin(q[1]) - model.l2 * sin(q[1] + q[2])
    pxq2 = -1.0 * model.l2 * sin(q[1] + q[2])
    pzq1 = model.l1 * cos(q[1]) + model.l2 * cos(q[1] + q[2])
    pzq2 = model.l2 * cos(q[1] + q[2])
    a = 2.0 * (px - x) * pxq1 + 2.0 * (pz - z) * pzq1
    b = 2.0 * (px - x) * pxq2 + 2.0 * (pz - z) * pzq2
    c = -2.0 * (px - x)
    d = -2.0 * (pz - z)

    @SMatrix [a b c d;
              0.0 0.0 0.0 1.0]
end

function P_func(model::RRParticle, q)
    a = -1.0 * model.l1 * sin(q[1]) - model.l2 * sin(q[1] + q[2])
    b = -1.0 * model.l2 * sin(q[1] + q[2])

    @SMatrix [a b -a -b;
              0.0 0.0 1.0 0.0
			  -a -b a b;
		      0.0 0.0 -1.0 0.0]
end

function friction_cone(model::RRParticle, u)
	λ = u[model.idx_λ]
	b = u[model.idx_b]
	return @SVector [model.μ * λ[1] - sum(b[1:2]),
	                 model.μ * λ[2] - sum(b[3:4])]
end

function maximum_dissipation(model::RRParticle, x⁺, u, h)
	q3 = x⁺[model.nq .+ (1:model.nq)]
	q2 = x⁺[1:model.nq]
	ψ = u[model.idx_ψ]
	ψ_stack = [ψ[1] * ones(2); ψ[2] * ones(2)]
	η = u[model.idx_η]
	return P_func(model, q3) * (q3 - q2) / h + ψ_stack - η
end

function fd(model::RRParticle{Discrete, FixedTime}, x⁺, x, u, w, h, t)
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
    + transpose(N_func(model, q3)) * SVector{2}(λ)
    + transpose(P_func(model, q3)) * SVector{4}(b)
    - h *(G_func(model, q2⁺) - 0.5 * C_func(model, q2⁺, q3, h)))]
end

model = RRParticle{Discrete, FixedTime}(n, m, d,
				   m1, J1, l1,
				   m2, J2, l2,
				   mp, μ, 9.81,
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

function visualize!(vis, model::RRParticle, q;
		Δt = 0.1, r = 0.1, cybertruck = true, x_offset = 0.5)

	default_background!(vis)

    l1 = Cylinder(Point3f0(0, 0, 0),
		Point3f0(0, 0, model.l1),
		convert(Float32, 0.025))
    setobject!(vis["l1"], l1, MeshPhongMaterial(color = RGBA(0, 0, 0, 1.0)))
    l2 = Cylinder(Point3f0(0, 0, 0), Point3f0(0, 0, model.l2),
		convert(Float32, 0.025))
    setobject!(vis["l2"], l2, MeshPhongMaterial(color = RGBA(0, 0, 0, 1.0)))

    setobject!(vis["elbow"], Sphere(Point3f0(0),
        convert(Float32, 0.05)),
        MeshPhongMaterial(color = RGBA(0, 0, 0, 1.0)))
    setobject!(vis["ee"], Sphere(Point3f0(0),
        convert(Float32, 0.05)),
        MeshPhongMaterial(color = RGBA(1.0, 165.0 / 255.0, 0.0, 1.0)))

    if cybertruck
        obj_path = joinpath(pwd(), "models/cybertruck/cybertruck.obj")
        mtl_path = joinpath(pwd(), "models/cybertruck/cybertruck.mtl")

        ctm = ModifiedMeshFileObject(obj_path, mtl_path, scale=0.07)
        setobject!(vis["particle"], ctm)
        settransform!(vis["particle"], compose(Translation([q[1][3] + x_offset,
			-1.0 * r, q[1][4]]), LinearMap(RotZ(pi) * RotX(pi / 2.0))))
    else
        setobject!(vis["particle"], Rect(Vec(0,0,0), Vec(2r,2r,2r)))
    end

    anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

    for t = 1:length(q)
        p_mid = [kinematics_mid(model,q[t])[1], 0.0, kinematics_mid(model, q[t])[2]]
        p_ee = [kinematics_ee(model, q[t])[1], 0.0, kinematics_ee(model, q[t])[2]]
        MeshCat.atframe(anim, t) do
            settransform!(vis["particle"], compose(Translation([q[t][3] + x_offset, -1.0 * r, q[t][4]]),
				LinearMap(RotZ(pi) * RotX(pi / 2.0))))

            settransform!(vis["l1"], cable_transform(zeros(3), p_mid))
            settransform!(vis["l2"], cable_transform(p_mid, p_ee))

            settransform!(vis["elbow"], Translation(p_mid))
            settransform!(vis["ee"], Translation(p_ee))
        end
    end

	settransform!(vis["/Cameras/default"],
	    compose(Translation(1.0, 0.0, -1.0),LinearMap(RotZ(-pi / 2.0))))

    MeshCat.setanimation!(vis, anim)
end
