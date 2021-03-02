"""
    box dynamics
    - 3D particle with orientation and 8 corners subject to contact forces

    - configuration: q = (px, py, pz, rx, ry, rz) ∈ R⁶
        - orientation : modified Rodrigues angles

    - contacts (8x)
        - impact force (magnitude): n ∈ R₊
        - friction force: b ∈ R²
            - contact force: λ = (b, n) ∈ R² × R₊
            - friction coefficient: μ ∈ R₊

    Discrete Mechanics and Variational Integrators
        pg. 363
"""

struct Box
	n::Int
	m::Int
	d::Int

    mass # mass
    J # inertia
    μ # friction coefficient
    g # gravity

    r             # corner length
	n_corners     # number of corners
    corner_offset # precomputed corner offsets

    nq # configurations dimension
end

# Methods
function mass_matrix(model::Box)
	Diagonal(@SVector [model.mass, model.mass, model.mass,
		model.J, model.J, model.J])
end

function gravity(model::Box)
	@SVector [0., 0., model.mass * model.g, 0., 0., 0.]
end

function kinematics(model::Box, q)
    p = view(q, 1:3)
    r = view(q, 4:6)

    R = MRP(r...)

    SVector{24}([(p + R * model.corner_offset[1])...,
                 (p + R * model.corner_offset[2])...,
                 (p + R * model.corner_offset[3])...,
                 (p + R * model.corner_offset[4])...,
                 (p + R * model.corner_offset[5])...,
                 (p + R * model.corner_offset[6])...,
                 (p + R * model.corner_offset[7])...,
                 (p + R * model.corner_offset[8])...])
end

function jacobian(model::Box, q)
	k(z) = kinematics(model, z)
	ForwardDiff.jacobian(k, q)
end

function signed_distance(model::Box, q)
	idx = collect([3, 6, 9, 12, 15, 18, 21, 24])
	kinematics(model, q)[idx]
end

function P_func(model::Box, q)
	idx = collect([1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23])
	k(z) = kinematics(model, z)[idx]
	ForwardDiff.jacobian(k, q)
end

function B_func(model::Box, q)
	# p = view(q, 1:3)
    r = view(q, 4:6)

    R = MRP(r...)
    # @SMatrix [0. 0. 0. R[1,1] R[2,1] R[3,1];
    #           0. 0. 0. R[1,2] R[2,2] R[3,2];
    #           0. 0. 0. R[1,3] R[2,3] R[3,3]]
	@SMatrix [1. 0. 0. 0.0 0.0 0.0;
			  0. 1. 0. 0.0 0.0 0.0;
			  0. 0. 1. 0.0 0.0 0.0;
			  0. 0. 0. R[1,1] R[2,1] R[3,1];
		      0. 0. 0. R[1,2] R[2,2] R[3,2];
		      0. 0. 0. R[1,3] R[2,3] R[3,3]]
end

# dynamics
function dynamics(model, q1, q2, q3, u1, λ, h)
      nq = model.nq
      SVector{nq}(mass_matrix(model) * (2.0 * q2 - q1 - q3) / h
            - h * gravity(model)
            + h * jacobian(model, q3)' * λ
			+ h * transpose(B_func(model, q3)) * u1)
end

# Kinematics
num_contacts = 8
d = 0.5
c8 = @SVector [d, d, d]
c2 = @SVector [d, d, -d]
c3 = @SVector [d, -d, d]
c4 = @SVector [d, -d, -d]
c5 = @SVector [-d, d, d]
c6 = @SVector [-d, d, -d]
c7 = @SVector [-d, -d, d]
c1 = @SVector [-d, -d, -d]

corner_offset = @SVector [c1, c2, c3, c4, c5, c6, c7, c8]

# Model
nq = 6
nu = 6
model = Box(2 * nq, nu, 0,
			1.0,
			1.0 / 12.0 * 1.0 * ((2.0 * d)^2 + (2.0 * d)^2),
 			0.1,
			9.81,
            d, num_contacts, corner_offset,
            nq)

# qq = rand(nq)
# kinematics(model, qq)
# ϕ_func(model, qq)
# jacobian(model, qq)
# P_func(model, qq)
num_contacts
# var
num_var = nq + num_contacts + num_contacts + 3 * num_contacts + 3 * num_contacts
function unpack(z)
	q = view(z, 1:nq)
	n = view(z, nq .+ (1:num_contacts))
	sϕ = view(z, nq + num_contacts .+ (1:num_contacts))
	b = view(z, nq + 2 * num_contacts .+ (1:3 * num_contacts))
	sb = view(z, nq + 2 * num_contacts + 3 * num_contacts .+ (1:3 * num_contacts))

	b_traj = [view(b, (i - 1) * 3 .+ (1:3)) for i = 1:num_contacts]
	sb_traj = [view(sb, (i - 1) * 3 .+ (1:3)) for i = 1:num_contacts]

	return q, n, sϕ, b_traj, sb_traj
end

function initialize(q2, num_var; z_init = 0.01)
	z = z_init * ones(num_var)

	z[1:nq] = copy(q2)
	z[nq + 2 * num_contacts .+ (1:3 * num_contacts)] = vcat([[1.0; 0.01; 0.01] for i = 1:num_contacts]...)
	z[nq + 2 * num_contacts + 3 * num_contacts .+ (1:3 * num_contacts)] = vcat([[1.0; 0.01; 0.01] for i = 1:num_contacts]...)

	return z
end

function visualize!(vis, model::Box, q;
        Δt = 0.1)

	default_background!(vis)

	setobject!(vis["box"], GeometryBasics.Rect(Vec(-1.0 * model.r,
		-1.0 * model.r,
		-1.0 * model.r),
		Vec(2.0 * model.r, 2.0 * model.r, 2.0 * model.r)),
		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

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
