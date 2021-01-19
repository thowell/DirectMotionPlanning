using LinearAlgebra, ForwardDiff, SparseArrays
using Convex, SCS

# problem setup
n = 2

_c = 1.0 * ones(n)
_b = 1.0

"Convex.jl"
x = Variable(n)
prob = minimize(_c' * x)
prob.constraints += norm(x) <= _b

optimizer = SCS.Optimizer()
@time solve!(prob, optimizer)

optimizer.cone
optimizer.sol

# cone program data
A = Matrix(SparseMatrixCSC{Float64, Int64}(optimizer.data.A.m,
    optimizer.data.A.n,
    optimizer.data.A.colptr,
    optimizer.data.A.rowval,
    optimizer.data.A.nzval))

b = optimizer.data.b
c = optimizer.data.c

m = optimizer.data.A.m
n = optimizer.data.A.n

# cone program solution
x = optimizer.data.primal
y = optimizer.data.dual
s = optimizer.data.slack

optimizer.data.dimension

optimizer.data.num_rows

prob.model.model.optimizer.data.primal

optimizer.data.primal
optimizer.data.dual

c
s' * y
# # cones
# function κ_free(z)
#     z
# end
#
# function κ_no(z)
#     return max.(0.0, z)
# end
#
# function κ_soc(z)
#     z1 = z[1:end-1]
#     z2 = z[end]
#
#     z_proj = zero(z)
#
#     if norm(z1) <= z2
#         z_proj = copy(z)
#     elseif norm(z1) <= -z2
#         z_proj = zero(z)
#     else
#         a = 0.5 * (1.0 + z2 / norm(z1))
#         z_proj[1:end-1] = a * z1
#         z_proj[end] = a * norm(z1)
#     end
#     return z_proj
# end
#
# function Pκ(z)
#     z_proj = zero(z)
#
#     z[1] = κ_free(z[1])
#     z[2] = κ_no(z[2])
#     z[3:5] = κ_soc(z[3:5])
#
#     z_proj
# end
#
# Pκ(s)
# s

b - A * x

# Q = Array([zeros(n, n) transpose(A) c;
#            -A zeros(m, m) b;
#            -c' -b'      0.0])
