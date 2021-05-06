# Model
include_model("hopper")

mb = 3.0 # body mass
ml = 0.3  # leg mass
Jb = 0.75 # body inertia
Jl = 0.075 # leg inertia

model = Hopper{Discrete, FixedTime}(n, m, d,
			   mb, ml, Jb, Jl,
			   0.1, g,
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

model_ft = free_time_model(model)

# Stair
function ϕ_func(model::Hopper, q)
	k = kinematics(model, q)
	if k[1] > 0.25
    	return @SVector [k[2] - 0.25]
	else
		return @SVector [k[2]]
	end
end

# Horizon
T = 26

# Time step
tf = 0.75
h = tf / (T - 1)

# Bounds
_uu = Inf * ones(model_ft.m)
_uu[model_ft.idx_u] .= Inf #model_ft.uU
_uu[end] = 2.0 * h
_ul = zeros(model_ft.m)
_ul[model_ft.idx_u] .= -Inf# model_ft.uL
_ul[end] = 0.5 * h
ul, uu = control_bounds(model_ft, T, _ul, _uu)

# Initial and final states
z_h = 0.25
q1 = [0.0, 0.5, 0.0, 0.5]
qM = [0.25, 0.5 + 2.0 * z_h, 0.0, 0.5]
qT = [0.5, 0.5 + z_h, 0.0, 0.5]
xl, xu = state_bounds(model_ft, T,
		[model_ft.qL; model_ft.qL],
		[model_ft.qU; model_ft.qU],
        x1 = [q1; q1],
		xT = [qT; qT])

# Objective
include_objective(["velocity", "nonlinear_stage", "control_velocity"])
qp = [0.01; 0.01; 1000.0; 1.0]
obj_tracking = quadratic_time_tracking_objective(
    [Diagonal(0.0 * [qp; qp]) for t = 1:T],
    [Diagonal([1.0e-2, 1.0e-2,
		1.0e-3 * ones(model_ft.nc)..., 1.0e-3 * ones(model_ft.nb)...,
		zeros(model_ft.m - model_ft.nu - model_ft.nc - model_ft.nb - 1)..., 0.0])
		for t = 1:T-1],
    [[qT; qT] for t = 1:T],
    [zeros(model_ft.m) for t = 1:T],
    1.0)

obj_contact_penalty = PenaltyObjective(1.0e5, model_ft.m - 1)

obj_velocity = velocity_objective(
    [Diagonal(1.0e-5 * ones(model_ft.nq)) for t = 1:T-1],
    model_ft.nq,
    h = h,
    idx_angle = collect([3]))


function l_foot_vel(x, u, t)
	J = 0.0

	q1 = view(x, 1:4)
	p1 = kinematics(model, q1)

	q2 = view(x, 4 .+ (1:4))
	p2 = kinematics(model, q2)

	v = (p2 - p1) ./ h

	if true
		J += 1000.0 * (q1 - q_ref[t])' * (q1 - q_ref[t])
	end

	if t < 8 || t > 18
		J += 1000.0 * v[1]^2.0
	end

	return J
end

l_foot_vel(x) = l_foot_vel(x, nothing, T)
obj_foot_vel = nonlinear_stage_objective(l_foot_vel, l_foot_vel)

obj_ctrl_vel = control_velocity_objective(Diagonal([1.0e-3 * ones(model_ft.nu);
	1.0e-3 * ones(model_ft.nc + model_ft.nb);
	zeros(model_ft.m - model_ft.nu - model_ft.nc - model_ft.nb)]))

obj = MultiObjective([obj_tracking, obj_contact_penalty, obj_velocity, obj_foot_vel, obj_ctrl_vel])

# Constraints
include_constraints(["free_time", "contact", "stage"])
con_free_time = free_time_constraints(T)
con_contact = contact_constraints(model_ft, T)

p1_ref = kinematics(model, q1)
pT_ref = kinematics(model, qT)

function pinned1!(c, x, u, t)
    q = view(x, 1:4)
    c[1:2] = p1_ref - kinematics(model, q)
	nothing
end

function pinnedT!(c, x, u, t)
    q = view(x, 4 .+ (1:4))
	c[1:2] = pT_ref - kinematics(model, q)
	nothing
end

T_fix = 5
n_stage = 2
t_idx1 = vcat([t for t = 1:T_fix]...)
t_idxT = vcat([(T - 2 * T_fix + 1):T]...)

con_pinned1 = stage_constraints(pinned1!, n_stage, (1:0), t_idx1)
con_pinnedT = stage_constraints(pinnedT!, n_stage, (1:0), t_idxT)
con = multiple_constraints([con_free_time, con_contact, con_pinned1, con_pinnedT])#, con_loop])

# Problem
prob = trajectory_optimization_problem(model_ft,
               obj,
               T,
               xl = xl,
               xu = xu,
               ul = ul,
               uu = uu,
               con = con)

ql1 = linear_interpolation(q1, qM, 14)
ql2 = linear_interpolation(qM, qT, 14)
q_ref = [ql1..., ql2[2:end]...]
# Trajectory initialization
x0 = configuration_to_state(q_ref) # linear interpolation on state
u0 = [[1.0e-3 * rand(model_ft.m-1); h] for t = 1:T-1] # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem
@time z̄, info = solve(prob, copy(z0),
	nlp = :ipopt,
	tol = 1.0e-4, c_tol = 1.0e-4, mapl = 5,
	time_limit = 60)
@show check_slack(z̄, prob)
x̄, ū = unpack(z̄, prob)
tf, t, h̄ = get_time(ū)

q = state_to_configuration(x̄)
u = [u[model.idx_u] for u in ū]
γ = [u[model.idx_λ] for u in ū]
b = [u[model.idx_b] for u in ū]
ψ = [u[model.idx_ψ] for u in ū]
η = [u[model.idx_η] for u in ū]
h̄ = mean(h̄)

# include(joinpath(pwd(), "models/visualize.jl"))
# vis = Visualizer()
# open(vis)
visualize!(vis, model_ft,
	q,
	Δt = h̄[1],
	scenario = :vertical)

hm = h̄
μm = model.μ
qm, um, γm, bm, ψm, ηm = q, u, γ, b, ψ, η

@save joinpath(@__DIR__, "hopper_stair.jld2") qm um γm bm ψm ηm μm hm
@load joinpath(@__DIR__, "hopper_stair.jld2") qm um γm bm ψm ηm μm hm

function step_repeat(q, u, γ, b, ψ, η, T; steps = 2)
	qm = [deepcopy(q)...]
	um = [deepcopy(u)...]
	γm = [deepcopy(γ)...]
	bm = [deepcopy(b)...]
	ψm = [deepcopy(ψ)...]
	ηm = [deepcopy(η)...]

	stride = zero(qm[1])
	for i = 1:(steps-1)
		@show stride[1] += q[T+1][1] - q[2][1]
		@show stride[2] += 0.25
		for t = 1:T-1
			push!(qm, q[t+2] + stride)
			push!(um, u[t])
			push!(γm, γ[t])
			push!(bm, b[t])
			push!(ψm, ψ[t])
			push!(ηm, η[t])
		end
	end

	return qm, um, γm, bm, ψm, ηm
end

qm, um, γm, bm, ψm, ηm = step_repeat(q, u, γ, b, ψ, η, T, steps = 3)

@save joinpath(@__DIR__, "hopper_stairs_3.jld2") qm um γm bm ψm ηm μm hm
@load joinpath(@__DIR__, "hopper_stairs_3.jld2") qm um γm bm ψm ηm μm hm

visualize!(vis, model_ft,
	qm,
	Δt = h̄[1],
	scenario = :stairs)

setobject!(vis["box1"], GeometryBasics.HyperRectangle(Vec(0.0, 0.0, 0.0),
	Vec(0.5, 0.5, 0.25)), MeshPhongMaterial(color = RGBA(0.5, 0.5, 0.5, 1.0)))
settransform!(vis["box1"], Translation(0.25, -0.25, 0))

setobject!(vis["box2"], GeometryBasics.HyperRectangle(Vec(0.0, 0.0, 0.0),
	Vec(0.5, 0.5, 2 * 0.25)), MeshPhongMaterial(color = RGBA(0.5, 0.5, 0.5, 1.0)))
settransform!(vis["box2"], Translation(0.25 + 0.5, -0.25, 0))

setobject!(vis["box3"], GeometryBasics.HyperRectangle(Vec(0.0, 0.0, 0.0),
	Vec(0.5, 0.5, 3 * 0.25)), MeshPhongMaterial(color = RGBA(0.5, 0.5, 0.5, 1.0)))
settransform!(vis["box3"], Translation(0.25 + 2 * 0.5, -0.25, 0))

tall_flip = load(joinpath(@__DIR__, "hopper_tall_flip.jld2"))

qm_f, um_f, γm_f, bm_f, ψm_f, ηm_f, μm_f, hm_f = tall_flip["qm"], tall_flip["um"], tall_flip["γm"], tall_flip["bm"], tall_flip["ψm"], tall_flip["ηm"], tall_flip["μm"], tall_flip["hm"]

str = zero(qm[1])
str[1] = qm[end][1]

for i = 1:10
	t = 1
	push!(qm, qm_f[t+2] + str)
	push!(um, um_f[t])
	push!(γm, γm_f[t])
	push!(bm, bm_f[t])
	push!(ψm, ψm_f[t])
	push!(ηm, ηm_f[t])
end

for t = 1:length(um_f)
	push!(qm, qm_f[t+2] + str)
	push!(um, um_f[t])
	push!(γm, γm_f[t])
	push!(bm, bm_f[t])
	push!(ψm, ψm_f[t])
	push!(ηm, ηm_f[t])
end

@save joinpath(@__DIR__, "hopper_stairs_3_flip.jld2") qm um γm bm ψm ηm μm hm


setprop!(vis["/Cameras/default/rotated/<object>"], "zoom", 20)

settransform!(vis["/Cameras/default"],
	compose(Translation(0.0, -90.0, -1.0),LinearMap(RotZ(-0.5 * π))))
