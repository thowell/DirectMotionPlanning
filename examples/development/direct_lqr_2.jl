n = 2
m = 1

A = [1.0 1.0; 0.0 1.0]
B = [0.0; 1.0]

Q = [1.0 0.0; 0.0 1.0]
R = [1.0]

# x_init = [1.0; 0.0]

nz = n + m + n

function unpack(z)
	x2 = z[1:n]
	u1 = z[n .+ (1:m)]
	y2 = z[n + m .+ (1:n)]

	return x2, u1, y2
end

z0 = rand(nz)

function lagrangian(z, x1)
	x2, u1, y2 = unpack(z)

	L = 0.0

	# objective
	L += transpose(x2) * Q * x2
	L += (transpose(u1) * R * u1)[1]

	# constraints
	L += transpose(y2) * (A * x1 + B * u1[1] - x2)

	return L
end

@variables z_sym[1:nz]
@variables x1_sym[1:n]

L = lagrangian(z_sym, x1_sym)
L = simplify.(L)

dL = Symbolics.gradient(L, z_sym)
ddL = Symbolics.hessian(L, z_sym)
ddLx1 = Symbolics.jacobian(dL, x1_sym)

L_grad = eval(Symbolics.build_function(dL, z_sym, x1_sym)[1])
L_hess = eval(Symbolics.build_function(ddL, z_sym, x1_sym)[1])
L_hess_x1 = eval(Symbolics.build_function(ddLx1, z_sym, x1_sym)[1])

function solve(z0, x1)
	z = copy(z0)

	∇L = L_grad(z, x1)

	println()
	for i = 1:10
		r_norm = norm(∇L)
		println("iter ($i)")
		println("	grad norm: $(r_norm)")
		r_norm < 1.0e-8 && break

		∇²L = L_hess(z, x1)

		Δ = ∇²L \ ∇L

		α = 1.0
		ẑ = z - α * Δ

		∇L̂ = L_grad(ẑ, x1)

		iter = 1
		while norm(∇L̂) > norm(∇L)
			α *= 0.5
			ẑ = z - α * Δ
			∇L̂ = L_grad(ẑ, x1)
			iter += 1

			if iter > 100
				break
			end
		end

		z = ẑ
		∇L = ∇L̂
	end

	return z, ∇L, L_hess(z, x1)
end

z_sol, ∇L_sol, ∇²L_sol = solve(z0, x1)
K_direct = (∇²L_sol \ L_hess_x1(z_sol, x1))[n .+ (1:m), :]

# LQR solution
K, P = tvlqr([A], [B], [Q, Q], [R[1,1]])

norm(K_direct - K[1])
