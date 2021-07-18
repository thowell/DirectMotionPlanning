function pack(X, U)
    z = zeros(num_var - p)
    for t = 1:T
        z[x_idx[t]] = X[t]
        t == T && continue
        z[u_idx[t]] = U[t]
    end
    return z
end

function unpack(z)
    x = [z[x_idx[t]] for t = 1:T]
    u = [z[u_idx[t]] for t = 1:T-1]
    return x, u
end

function primal_bounds(x1, xT)
    zl = -Inf * ones(num_var)
    zu = Inf * ones(num_var)

	zl[x_idx[1]] = x1
	zu[x_idx[1]] = x1

	# zl[x_idx[T]] = xT
	# zu[x_idx[T]] = xT

    return zl, zu
end

function constraint_bounds()
    cl = zeros(num_con)
    cu = zeros(num_con)
    return cl, cu
end

function eval_objective(prob::MOIGeneralProblem, z)
	J = 0.0
	for t = 1:T
		if t < T
			x = z[x_idx[t]]
			u = z[u_idx[t]]
			J += objective(x, u, t)
		else
			x = z[x_idx[t]]
			J += objective(x, nothing, t)
		end
	end
	# parameter regularization
	θ = z[p_idx]
	J += 1.0e-5 * θ' * θ
	return J
end

function eval_objective_gradient!(∇J, z, prob::MOIGeneralProblem)
    ∇J .= 0.0
	for t = 1:T
		if t < T
			x = z[x_idx[t]]
			u = z[u_idx[t]]
			obj_x(a) = objective(a, u, t)
			obj_u(a) = objective(x, a, t)
			∇J[x_idx[t]] = ForwardDiff.gradient(obj_x, x)
			∇J[u_idx[t]] = ForwardDiff.gradient(obj_u, u)
		else
			x = z[x_idx[t]]
			obj_xT(a) = objective(a, nothing, t)
			∇J[x_idx[t]] = ForwardDiff.gradient(obj_xT, x)
		end
	end
	θ = z[p_idx]
	∇J[p_idx] = 2.0 * 1.0e-5 * θ

    return nothing
end

function eval_constraint!(c, z, prob::MOIGeneralProblem)
	# dynamics
    for t = 1:T-1
		x = z[x_idx[t]]
		u = z[u_idx[t]]
		y = z[x_idx[t+1]]
		c[n_idx[t]] = dynamics(y, x, u, t)
	end

	# policy
	θ = z[p_idx]
	for t = 1:T-1
		x = z[x_idx[t]]
		u = z[u_idx[t]]
		c[num_dyn .+ m_idx[t]] = u - policy(x, θ)[1]
	end
    return nothing
end

function eval_constraint_jacobian!(∇c, z, prob::MOIGeneralProblem)
	shift = 0

    for t = 1:T-1
        x = view(z, x_idx[t])
        u = view(z, u_idx[t])
        y = view(z, x_idx[t + 1])

        dyn_x(a) = dynamics(y, a, u, t)
        dyn_u(a) = dynamics(y, x, a, t)
        dyn_y(a) = dynamics(a, x, u, t)

        r_idx = n_idx[t]

        s = n * n
        ∇c[shift .+ (1:s)] = vec(ForwardDiff.jacobian(dyn_x, x))
        shift += s

        s = n * m
        ∇c[shift .+ (1:s)] = vec(ForwardDiff.jacobian(dyn_u, u))
        shift += s

        s = n * n
        ∇c[shift .+ (1:s)] = vec(ForwardDiff.jacobian(dyn_y, y))
        shift += s
    end

	# policy
	θ = z[p_idx]
	for t = 1:T-1
		x = z[x_idx[t]]
		u = z[u_idx[t]]

		px, pθ = policy_gradients(x, θ)

		r_idx = num_dyn .+ m_idx[t]

		s = m * m
		∇c[shift .+ (1:s)] = vec(Diagonal(ones(m)))
		shift += s

		s = m * n
		∇c[shift .+ (1:s)] = -vec(px)
		shift += s

		s = m * p
		∇c[shift .+ (1:s)] = -vec(pθ)
		shift += s
	end

    return nothing
end

function sparsity_jacobian(prob::MOIGeneralProblem)

	row = []
	col = []

	for t = 1:T-1
		r_idx = n_idx[t]
		row_col!(row, col, r_idx, x_idx[t])
		row_col!(row, col, r_idx, u_idx[t])
		row_col!(row, col, r_idx, x_idx[t + 1])
	end

	for t = 1:T-1
		r_idx = num_dyn .+ (m_idx[t])
		row_col!(row, col, r_idx, u_idx[t])
		row_col!(row, col, r_idx, x_idx[t])
		row_col!(row, col, r_idx, p_idx)
	end

	return collect(zip(row, col))
end

## Trajectory Optimization

# time
T = 11
h = 1.0

# dynamics
n = 2
m = 1

A = SMatrix{n,n}([1.0 1.0; 0.0 1.0])
B = SMatrix{n,m}([0.0; 1.0])

function dynamics(y, x, u, t)
	y - (A * x + B * u)
end

# objective
function objective(x, u, t)
	J = 0.0
	if t == T
		Q = Diagonal(100.0 * ones(n))
		J += x' * Q  * x
	else
		Q = Diagonal(ones(n))
		R = Diagonal(0.1 * ones(m))
		J += x' * Q  * x
		J += transpose(u) * R * u
	end
	return J
end

## policy

T_mpc = 5
n = 2
m = 1

x_init = x1
_Q = sqrt.(ones(n))
_R = sqrt.(1.0 * ones(m))
_QT = sqrt.(1.0 * ones(n))
x_final = xT

nθ_mpc = n + n + m + n + n

num_x_mpc = n * T_mpc
num_u_mpc = m * (T_mpc - 1)
num_y_mpc = n * T_mpc
nz_mpc = num_x_mpc + num_u_mpc + num_y_mpc

x_idx_mpc = [collect((t - 1) * n .+ (1:n)) for t = 1:T_mpc]
u_idx_mpc = [collect(num_x_mpc + (t - 1) * m .+ (1:m)) for t = 1:T_mpc-1]
y_idx_mpc = [collect(num_x_mpc + num_u_mpc + (t - 1) * n .+ (1:n)) for t = 1:T_mpc]

x_mpc = linear_interpolation(x_init, x_final, T_mpc)
u_mpc = [0.01 * randn(m) for t = 1:T_mpc-1]
y_mpc = [zeros(n) for t = 1:T_mpc]

z0 = [vcat(x_mpc...); vcat(u_mpc...); vcat(y_mpc...)]
θ0 = [x_init; _Q; _R; _QT; x_final]

function lagrangian(z, θ)
	x_init = θ[1:n]
	_Q = θ[n .+ (1:n)]
	Qt = Diagonal(_Q.^2.0)
	_R = θ[n + n .+ (1:m)]
	Rt = Diagonal(_R.^2.0)
	_QT = θ[n + n + m .+ (1:n)]
	QT = Diagonal(_QT.^2.0)
	x_final = θ[n + n + m + n .+ (1:n)]

	L = 0.0

	x1 = view(z, x_idx_mpc[1])
	y1 = view(z, y_idx_mpc[1])

	L += transpose(y1) * (x_init - x1)

	for t = 1:T_mpc-1
		xt = view(z, x_idx_mpc[t])
		ut = view(z, u_idx_mpc[t])
		yt⁺ = view(z, y_idx_mpc[t+1])
		xt⁺ = view(z, x_idx_mpc[t+1])

		L += transpose(xt - x_final) * Qt * (xt - x_final)
		L += (transpose(ut) * Rt * ut)

		L += transpose(yt⁺) *  dynamics(xt⁺, xt, ut, t)
	end

	xT = view(z, x_idx_mpc[T_mpc])
	# yT⁺ = view(z, y_idx_mpc[T+1])

	L += transpose(xT - x_final) * QT * (xT - x_final)
	# L += transpose(yT⁺) * (x_final - xT)

	return L
end

lagrangian(z0, θ0)

@variables z_sym[1:nz_mpc]
@variables θ_sym[1:nθ_mpc]
@variables κ_sym[1:1]

L = lagrangian(z_sym, θ_sym);
L = simplify.(L);

dL = Symbolics.gradient(L, z_sym)
ddL = Symbolics.sparsehessian(L, z_sym)

L_grad = eval(Symbolics.build_function(dL, z_sym, θ_sym, κ_sym)[1])
L_hess = eval(Symbolics.build_function(ddL, z_sym, θ_sym)[1])

L_grad! = eval(Symbolics.build_function(dL, z_sym, θ_sym, κ_sym)[2])
L_hess! = eval(Symbolics.build_function(ddL, z_sym, θ_sym)[2])

L_grad(z0, θ0, 1.0)
L_hess(z0, θ0)

dLθ = Symbolics.jacobian(dL, θ_sym)

dL_jacθ = eval(Symbolics.build_function(dLθ, z_sym, θ_sym)[1])
dL_jacθ! = eval(Symbolics.build_function(dLθ, z_sym, θ_sym)[2])

dL_jacθ(z0, θ0)

# options
opts = InteriorPointOptions(
	κ_init = 1.0,
	κ_tol = 1.0,
	r_tol = 1.0e-8,
	diff_sol = true)

# solver
ip = interior_point(z0, θ0,
	r! = L_grad!, rz! = L_hess!,
	rz = similar(ddL, Float64),
	rθ! = dL_jacθ!,
	rθ = similar(dLθ, Float64),
	opts = opts)

# solve
status = interior_point!(ip)

using Test
# test
@test status
@test norm(ip.z[x_idx_mpc[1]] - x_init) < 1.0e-8

norm(ip.z[x_idx_mpc[T_mpc]] - x_final)

x_traj_mpc = [ip.z[x_idx_mpc[t]] for t = 1:T_mpc]
u_traj_mpc = [ip.z[u_idx_mpc[t]] for t = 1:T_mpc-1]

plot(hcat(x_traj_mpc...)')
plot(hcat(u_traj_mpc...)')

function policy(x, θ)
	x_mpc = linear_interpolation(x, x_final, T_mpc)
	u_mpc = [0.01 * randn(m) for t = 1:T_mpc-1]
	y_mpc = [zeros(n) for t = 1:T_mpc]

	# initialized solver primals
	ip.z .= copy([vcat(x_mpc...); vcat(u_mpc...); vcat(y_mpc...)])

	# initialized solver parameters
	_Q = θ[1:n]
	_R = θ[n .+ (1:m)]
	_QT = θ[n + m .+ (1:n)]
	ip.θ .= copy([x; _Q; _R; _QT; x_final])

	status = interior_point!(ip)

	!status && (@warn "solver failure")

	return ip.z[u_idx_mpc[1]], ip
	# return reshape(θ, m, n) * x, nothing
end

function policy_gradients(x, θ)
	_, ip = policy(x, θ)

	ux = ip.δz[u_idx_mpc[1], 1:n]
	uθ = ip.δz[u_idx_mpc[1], n .+ (1:(n + m + n))]

	return ux, uθ

	# px(a) = policy(a, θ)[1]
	# pθ(a) = policy(x, a)[1]
	# return ForwardDiff.jacobian(px, x), ForwardDiff.jacobian(pθ, θ)
end

# policy(x1, 0.01 * randn(p))
# policy_gradients(x1, 0.01 * randn(p))
θ0_policy = [_Q; _R; _QT]
policy(x1, θ0_policy)[1]
policy_gradients(x1, θ0_policy)

# p = m * n
p = n + m + n

# trajectory indices
x_idx = [(t - 1) * (n + m) .+ (1:n) for t = 1:T]
u_idx = [(t - 1) * (n + m) + n .+ (1:m) for t = 1:T-1]
n_idx = [(t - 1) * n .+ (1:n) for t = 1:T-1]
m_idx = [(t - 1) * m .+ (1:m) for t = 1:T-1]
p_idx = n * T + m * (T - 1) .+ (1:p)

# MOI setup
num_x = n * T
num_u = m * (T - 1)
num_var = num_x + num_u + p

num_dyn = n * (T - 1)
num_policy = m * (T - 1)
num_con = num_dyn + num_policy

# initialization
x1 = [1.0; 0.0]
xT = [0.0; 0.0]

U = [0.1 * randn(m) for t = 1:T-1]

# rollout
x_hist = [copy(x1)]
u_hist = []
Js = 0.0
for t = 1:T-1
	push!(u_hist, policy(x_hist[end], θ0_policy)[1])
	# push!(u_hist, policy(x_hist[end], θ_sol)[1])

	Js += objective(x_hist[end], u_hist[end], t)
	push!(x_hist, A * x_hist[end] + B * u_hist[end])
end
Js += objective(x_hist[end], nothing, T)
@show Js
@show norm(x_hist[end] - xT)


z = [pack(x_hist, u_hist); θ0_policy]
# z = [pack(X, U); 0.01 * randn(p)]# _Q; _R; _QT]

prob = MOIGeneralProblem(num_var, num_con, primal_bounds(x1, xT), constraint_bounds())
prob_moi = moi_problem(prob)

z_sol, info = solve(prob_moi, z,
		tol = 1.0e-3,
		c_tol = 1.0e-2,
		max_iter = 1000,
		nlp = :ipopt)

x_sol, u_sol = unpack(z_sol)
θ_sol = z_sol[p_idx]

norm(θ_sol - θ0_policy)
# M = reshape(θ_sol, m, n)

using Plots

plot(hcat(x_sol...)')
plot!(hcat(x_hist...)')
plot(hcat(u_sol..., u_sol[end])', linetype = :steppost)
plot!(hcat(u_hist..., u_hist[end])', linetype = :steppost)

## simulate policy
x1_sim = copy(x1) + 0.0 * randn(n)

x_hist = [x1_sim]
u_hist = []
Js = 0.0
for t = 1:T-1
	# push!(u_hist, policy(x_hist[end], θ0_policy)[1])
	push!(u_hist, policy(x_hist[end], θ_sol)[1])

	Js += objective(x_hist[end], u_hist[end], t)
	push!(x_hist, A * x_hist[end] + B * u_hist[end])
end
Js += objective(x_hist[end], nothing, T)
@show Js
plot(hcat(x_hist...)')
plot(hcat(u_hist..., u_hist[end])', linetype = :steppost)

@show norm(x_hist[end] - xT)
Q_sol = Diagonal(θ_sol[1:n].^2.0)
R_sol = Diagonal(θ_sol[n .+ (1:m)].^2.0)
QT_sol = Diagonal(θ_sol[n + m .+ (1:n)].^2.0)
