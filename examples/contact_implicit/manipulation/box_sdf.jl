# rotation matrix (2D)
function rotation_matrix(x)
	[cos(x) -sin(x); sin(x) cos(x)]
end

# signed distance for a box
function sd_box(p, dim)
	q = abs.(p) - dim
	norm(max.(q, 0.0)) + min(maximum(q), 0.0)
end

function sd_2d_box(p, pose, dim, rnd)
	x, y, θ = pose
	R = rotation_matrix(-θ)
	p_rot = R * (p - pose[1:2])

	return sd_box(p_rot, dim) - rnd
end

# box parameters
dim = [1.0, 1.0]
rnd = 0.01
dim_rnd = dim .- rnd

# setup
p = [0.5 * sqrt(2.0), 0.5 * sqrt(2.0)]
px = 0.0
py = 0.0
θ = 0.25 * π
pose = [px, py, θ]

# problem
@show sdf = sd_2d_box(p, pose, dim_rnd, rnd)

sd_pose(x) = sd_2d_box(p, x, dim_rnd, rnd)
@show Npose = ForwardDiff.gradient(sd_pose, pose)

sd_p(x) = sd_2d_box(x, pose, dim_rnd, rnd)
@show Np = ForwardDiff.gradient(sd_p, p)

n_dir = Np[1:2] ./ norm(Np[1:2])
t_dir = rotation_matrix(0.5 * π) * n_dir

r = p - pose[1:2]
m = cross([r; 0.0], [t_dir; 0.0])[3]

@show Ppose = [t_dir[1], t_dir[2], cross([p; 0.0] - [pose[1:2]; 0.0], [t_dir; 0.0])[3]]

@show Pp = -t_dir
