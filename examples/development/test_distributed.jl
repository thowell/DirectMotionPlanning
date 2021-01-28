using LinearAlgebra, BenchmarkTools
using Distributed, SharedArrays

@everywhere N = 1000
@everywhere n = 1000
@everywhere W = [rand(n, n) for i = 1:N]
@everywhere A = [W[i]' * W[i] for i = 1:N]
@everywhere b = [rand(n) for i = 1:N]

# Distributed
addprocs(4)
procs()
workers()
nworkers()
myid()
#
# remotecall_fetch(() -> myid(), 3)
# pmap(i -> println("I'm worker $(myid()), working on i=$i"), 1:10)
# @sync @distributed for i in 1:10
#   println("I'm worker $(myid()), working on i=$i")
# end
#
# a = Array{Int}(undef, 10)
# a
# @sync @distributed for i in 1:10
#   println("working on i=$i")
#   a[i] = i^2
# end
# a
#

# @everywhere printsquare(i) = println("working on i=$i: its square it $(i^2)")
# @sync @distributed for i in 1:10
#   printsquare(i)
# end

function solve_dist(x, A, b, N)
    @sync @distributed for i = 1:N
        x[i] = A[i] \ b[i]
    end
end

@everywhere x = SharedArray{Float64}((n, N))

print("run distributed")
solve_dist(x, A, b, N)
