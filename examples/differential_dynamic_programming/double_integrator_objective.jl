using Plots
using Random
Random.seed!(1)

include_ddp()

# Model
include_model("double_integrator")

p = 2 * model.n + 2 * model.m + 2 * model.n

function f(model::DoubleIntegratorContinuous, x, u, w)
    [x[2]; u[1]]
end

function fd(model::DoubleIntegratorContinuous{Midpoint, FixedTime}, x, u, w, h, t)
	if t == 1
		θ = u[1 .+ (1:p)]
    	return [x + h * f(model, x + 0.5 * h * f(model, x, view(u, 1:1), w), view(u, 1:1), w); θ]
	else
		θ = x[2 .+ (1:p)]
		return [view(x, 1:2) + h * f(model, view(x, 1:2) + 0.5 * h * f(model, view(x, 1:2), u, w), u, w); θ]
	end
end

# Time
T = 11
h = 0.1

model = DoubleIntegratorContinuous{Midpoint, FixedTime}(2, 1, 0)
n = [t == 1 ? model.n : model.n + p for t = 1:T]
m = [t == 1 ? model.m + p : model.m for t = 1:T]

# Initial conditions, controls, disturbances
x1 = [0.0; 0.0]
xT = [1.0; 0.0]

Q10 = ones(model.n)
q10 = randn(model.n)
R10 = ones(model.m)
r10 = randn(model.m)
QT0 = ones(model.n)
qT0 = randn(model.n)
θ0 = [Q10; q10; R10; r10; QT0; qT0]
# θ0 = [1.0]
ū = [t == 1 ? [0.01 * randn(1); θ0] : 0.01 * randn(1) for t = 1:T-1]
w = [zeros(model.d) for t = 1:T-1]

# Rollout
x̄ = rollout(model, x1, ū, w, h, T)

# Objective

obj = StageCosts([NonlinearCost() for t = 1:T], T)

function g(obj::StageCosts, x, u, t)
	T = obj.T
    if t < T
		t == 1 ? (θ = u[1 .+ (1:p)]) : (θ = x[2 .+ (1:p)])

		Q = Diagonal(θ[1:model.n])
		q = θ[model.n .+ (1:model.n)]
		R = Diagonal(θ[2 * model.n .+ (1:model.m)])
		r = θ[2 * model.n + model.m .+ (1:model.m)]

        return 1.0e-5 * u' * u #+ view(x, 1:model.n)' * Q * view(x, 1:model.n) + q' * view(x, 1:model.n) + view(u, 1:model.m)' * R * view(u, 1:model.m) + r' * view(u, 1:model.m)
    elseif t == T
		θ = x[2 .+ (1:p)]

		Q = Diagonal(θ[2 * model.n + 2 * model.m .+ (1:model.n)])
		q = θ[2 * model.n + 2 * model.m + model.n .+ (1:model.n)]

        return view(x, 1:model.n)' * Q * view(x, 1:model.n) + q' * view(x, 1:model.n)
    else
        return 0.0
    end
end

g(obj, x̄[1], ū[1], 1)
g(obj, x̄[2], ū[2], 2)
g(obj, x̄[T], nothing, T)
objective(obj, x̄, ū)

# Constraints
p_con = [t == 1 ? 0 : (t == T ? model.n : 0) for t = 1:T]
info_1 = Dict()#):x1l => [0.0; -1.0], :x1u => [0.0; 1.0], :inequality => (1:2 * model.n))
info_M = Dict()#:xM => xM)
info_t = Dict()
info_T = Dict(:xT => xT)
con_set = [StageConstraint(p_con[t], t < T ? (t == 1 ? info_1 : info_t) : info_T) for t = 1:T]

function c!(c, cons::StageConstraints, x, u, t)
	T = cons.T
	# if t == 1
	# 	c[1:2] .= view(u, 2:3) - cons.con[t].info[:x1u]
	# 	c[3:4] = cons.con[t].info[:x1l] - view(u, 2:3)
	# end
	# if t == Tm
	# 	c[1:2] .= view(x, 1:2) - cons.con[t].info[:xM]
	# end
	if t == T
		c[1:2] .= view(x, 1:2) - cons.con[t].info[:xT]
	end
end

prob = problem_data(model, obj, con_set, copy(x̄), copy(ū), w, h, T, n = n, m = m)
prob.m_data.obj
# objective_derivatives!(prob.m_data.obj, prob.m_data)

t = T
gx(z) = g(obj, z, t == T ? nothing : ū[t], t)
gu(z) = g(obj, x̄[t], z, t)
gz(z) = g(obj, z[1:n[t]], z[n[t] .+ (1:m[t])], t)

data = prob.m_data
ForwardDiff.gradient!(data.obj_deriv.gx[t], gx, x̄[t])
ForwardDiff.gradient!(data.obj_deriv.gu[t], gu, ū[t])
# ForwardDiff.hessian!(data.obj_deriv.gxx[t], gx, x̄[t])
using Symbolics
@variables x_sym[1:n[t]]

gx_sym = g(obj, x_sym, nothing, t)
gx_sym = simplify.(gx_sym)
dd_gx_sym = Symbolics.hessian(gx_sym, x_sym, simplify = true)

dd_gx_expr = Symbolics.build_function(dd_gx_sym, x_sym)[1]
dd_gx = eval(dd_gx_expr)
show(dd_gx(x̄[t]))

ForwardDiff.hessian!(data.obj_deriv.guu[t], gu, ū[t])
data.obj_deriv.gux[t] .= ForwardDiff.hessian(gz,
	[x̄[t]; ū[t]])[n[t] .+ (1:m[t]), 1:n[t]]

# Solve
@time constrained_ddp_solve!(prob,
    max_iter = 1000, max_al_iter = 10,
	ρ_init = 1.0, ρ_scale = 10.0)

# # Solve
# @time ddp_solve!(prob,
#     max_iter = 100, verbose = true)

x, u = current_trajectory(prob)
x̄, ū = nominal_trajectory(prob)

# x[1] = u[1][2:3]
# x̄[1] = ū[1][2:3]
# norm(x[1] - x[T][1:2])

# Visualize
using Plots
plot(hcat([xt[1:2] for xt in x]...)[:, 1:T]', color = :black, label = "")
# plot(hcat(u..., u[end])', linetype = :steppost)
