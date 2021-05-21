T = 3
n = 2
m = 1

A = [1.0 1.0; 0.0 1.0]
B = [0.0; 1.0]

Q = [1.0 0.0; 0.0 1.0]
R = [1.0]

x_init = [1.0; 0.0]

nz = T * n + (T - 1) * m + T * n

function unpack(z)
	x3 = z[1:n]
	u2 = z[n .+ (1:m)]
	x2 = z[n + m .+ (1:n)]
	u1 = z[n + m + n .+ (1:m)]
	x1 = z[n + m + n + m .+ (1:n)]
	y3 = z[n + m + n + m + n .+ (1:n)]
	y2 = z[n + m + n + m + n + n .+ (1:n)]
	y1 = z[n + m + n + m + n + n + n .+ (1:n)]

	return x3, u2, x2, u1, x1, y3, y2, y1
end

z0 = rand(nz)

function lagrangian(z, x_init)
	x3, u2, x2, u1, x1, y3, y2, y1 = unpack(z)

	L = 0.0

	# objective
	L += transpose(x3) * Q * x3
	L += (transpose(u2) * R * u2)[1]
	L += transpose(x2) * Q * x2
	L += (transpose(u1) * R * u1)[1]
	L += transpose(x1) * Q * x1

	# constraints
	L += transpose(y3) * (A * x2 + B * u2[1] - x3)
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
ddLx = Symbolics.jacobian(dL, x_sym)

L_grad = eval(Symbolics.build_function(dL, z_sym, x_sym)[1])
L_hess = eval(Symbolics.build_function(ddL, z_sym, x_sym)[1])
Lx_hess = eval(Symbolics.build_function(ddLx, z_sym, x_sym)[1])

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
K_direct = (∇²L_sol \ Lx_hess(z_sol, x_init))[n + m + n .+ (1:m), :]

# LQR solution
K, P = tvlqr([A for t = 1:T-1], [B for t = 1:T-1], [Q for t = 1:T], [R[1,1] for t = 1:T-1])

@assert norm(K[1] - K_direct) < 1.0e-8
