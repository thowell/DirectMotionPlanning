# Model
include_model("particle")

# Dimensions
nq = 3                    # configuration dimension
nu = 3                    # control dimension
nc = 1                    # number of contact points
nf = 2                    # number of parameters for friction cone
nb = nc * nf
ns = 1

# Parameters
μ = 0.5      # coefficient of friction
mass = 1.0   # mass
g = 9.81

n = 2 * nq
m = nu + nc + nb + 3 + ns
d = 0

idx_u = (1:nu)
idx_λ = nu .+ (1:nc)
idx_b = nu + nc .+ (1:nb)
idx_ψ = nu + nc + nb .+ (1:3)
idx_η = (1:0)#nu + nc + nb + nc .+ (1:nb)
idx_s = nu + nc + nb + 3 .+ (1:ns)

# Methods
function M_func(model::Particle, q)
    @SMatrix [model.mass 0.0 0.0;
              0.0 model.mass 0.0;
              0.0 0.0 model.mass]
end

G_func(model::Particle, q) = @SVector [0.0, 0.0, model.mass * model.g]

function ϕ_func(::Particle, q)
    @SVector[q[3]]
end

B_func(::Particle, q) = @SMatrix [1.0 0.0 0.0;
                                  0.0 1.0 0.0;
                                  0.0 0.0 1.0]

N_func(::Particle, q) = @SMatrix [0.0 0.0 1.0]

function P_func(model::Particle, q)
   return @SMatrix [1.0 0.0 0.0;
                    0.0 1.0 0.0]
end

# function P_func(model::Particle, q)
#    return @SMatrix [1.0 0.0 0.0;
#                     0.0 1.0 0.0;
#                     -1.0 0.0 0.0;
#                     0.0 -1.0 0.0]
# end

function friction_cone(model::Particle, u)
	λ = u[model.idx_λ]
	b = u[model.idx_b]
	return @SVector [model.μ * λ[1] - norm(b' * b)]
end

function maximum_dissipation(model::Particle, x⁺, u, h)
	q3 = x⁺[model.nq .+ (1:model.nq)]
	q2 = x⁺[1:model.nq]

	λ = u[model.idx_λ]
	b = u[model.idx_b]
	ψ = u[model.idx_ψ]

	return [P_func(model, q3) * (q3 - q2) / h - ψ[2:3];
		model.μ * λ[1] * ψ[2:3] + ψ[1] * b]
end

function fd(model::Particle{Discrete, FixedTime}, x⁺, x, u, w, h, t)
	q3 = view(x⁺, model.nq .+ (1:model.nq))
	q2⁺ = view(x⁺, 1:model.nq)
	q2⁻ = view(x, model.nq .+ (1:model.nq))
	q1 = view(x, 1:model.nq)
	u_ctrl = view(u, model.idx_u)
	λ = view(u, model.idx_λ)
	b = view(u, model.idx_b)

    [q2⁺ - q2⁻;
    ((1.0 / h) * (M_func(model, q1) * (SVector{3}(q2⁺) - SVector{3}(q1))
    - M_func(model, q2⁺) * (SVector{3}(q3) - SVector{3}(q2⁺)))
    + h * (transpose(B_func(model, q3)) * SVector{3}(u_ctrl)
    + transpose(N_func(model, q3)) * SVector{1}(λ)
    + transpose(P_func(model, q3)) * SVector{2}(b)
    - G_func(model, q2⁺)))]
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


# Horizon
T = 11

# Time step
tf = 1.0
h = tf / (T-1)

# Bounds
_uu = Inf * ones(model.m)
_uu[model.idx_u] .= 0.0
_ul = zeros(model.m)
_ul[model.idx_b] .= -Inf
_ul[model.idx_ψ] .= -Inf
ul, uu = control_bounds(model, T, _ul, _uu)

# Initial and final states
q1 = [0.0, 0.0, 1.0]
v1 = [1.0, 1.0, 0.0]
v2 = v1 - G_func(model,q1) * h
q2 = q1 + 0.5 * h * (v1 + v2)

x1 = [q1; q2]

xl, xu = state_bounds(model, T, x1 = x1)

# Objective
obj = PenaltyObjective(1.0e3, model.m)

# Constraints
include_constraints("contact_soc")
con_contact = contact_constraints(model, T)

# Problem
prob = trajectory_optimization_problem(model,
               obj,
               T,
               h = h,
               xl = xl,
               xu = xu,
               ul = ul,
               uu = uu,
               con = con_contact)

# Trajectory initialization
x0 = [0.01 * rand(model.n) for t = 1:T] #linear_interpolation(x1, x1, T) # linear interpolation on state
u0 = [0.001 * rand(model.m) for t = 1:T-1] # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem
@time z̄, info = solve(prob, copy(z0), tol = 1.0e-5, c_tol = 1.0e-5)

check_slack(z̄, prob)
x̄, ū = unpack(z̄, prob)

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
visualize!(vis, model,
    state_to_configuration(x̄),
    Δt = h)

open(vis)
