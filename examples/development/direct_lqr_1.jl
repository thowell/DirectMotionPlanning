n = 2
m = 1

A = [1.0 1.0; 0.0 1.0]
B = [0.0; 1.0]

Q = [1.0 0.0; 0.0 1.0]
R = [1.0]

x_init = [1.0; 0.0]

nz = 2 * n + m + 2 * n

function unpack(z)
	x2 = z[1:n]
	u1 = z[n .+ (1:m)]
	x1 = z[n + m .+ (1:n)]
	y2 = z[n + m + n .+ (1:n)]
	y1 = z[n + m + n + n .+ (1:n)]

	return x2, u1, x1, y2, y1
end

z0 = rand(nz)
unpack(z0)
ones(1)' * R * ones(1)
function lagrangian(z, x_init)
	x2, u1, x1, y2, y1 = unpack(z)

	L = 0.0

	# objective
	L += transpose(x2) * Q * x2
	L += (transpose(u1) * R * u1)[1]
	L += transpose(x1) * Q * x1

	# constraints
	L += transpose(y2) * (A * x1 + B * u1[1] - x2)
	L += transpose(y1) * (x_init - x1)

	return L
end

@variables z_sym[1:nz]
@variables x_sym[1:n]

L = lagrangian(z_sym, x_sym)
L = simplify.(L)

dL = Symbolics.gradient(L, z_sym)
ddL = Symbolics.hessian(L, z_sym)
ddL_x = Symbolics.jacobian(dL, x_sym)

L_grad = eval(Symbolics.build_function(dL, z_sym, x_sym)[1])
L_hess = eval(Symbolics.build_function(ddL, z_sym, x_sym)[1])
Lx_hess = eval(Symbolics.build_function(ddL_x, z_sym, x_sym)[1])

function solve(z0, x_init)
	z = copy(z0)

	∇L = L_grad(z, x_init)

	println()
	for i = 1:10
		r_norm = norm(∇L)
		println("iter ($i)")
		println("	grad norm: $(r_norm)")
		r_norm < 1.0e-8 && break

		∇²L = L_hess(z, x_init)

		Δ = ∇²L \ ∇L

		α = 1.0
		ẑ = z - α * Δ

		∇L̂ = L_grad(ẑ, x_init)

		iter = 1
		while norm(∇L̂) > norm(∇L)
			α *= 0.5
			ẑ = z - α * Δ
			∇L̂ = L_grad(ẑ, x_init)
			iter += 1

			if iter > 100
				break
			end
		end

		z = ẑ
		∇L = ∇L̂
	end

	return z, ∇L, L_hess(z, x_init)
end

z_sol, ∇L_sol, ∇²L_sol = solve(z0, x_init)
K_direct = (∇²L_sol \ Lx_hess(z_sol, x_init))[n .+ (1:m), :]

# LQR solution
K, P = tvlqr([A], [B], [Q, Q], [R[1,1]])

@assert norm(K_direct - K[1]) < 1.0e-8
