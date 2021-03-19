"""
    test trajectory optimization
"""
prob_traj = prob.prob

# objective
for i = 1:1
    z0 = rand(prob_traj.num_var)
    tmp_o(z) = eval_objective(prob_traj, z)
    ∇j = zeros(prob_traj.num_var)
    eval_objective_gradient!(∇j, z0, prob_traj)
    @assert norm(ForwardDiff.gradient(tmp_o, z0) - ∇j, Inf) < 1.0e-5
    println("i $i")
end

# constraints
c0 = zeros(prob_traj.num_con)
eval_constraint!(c0, z0, prob_traj)
tmp_c(c, z) = eval_constraint!(c, z, prob_traj)
∇c_fd = ForwardDiff.jacobian(tmp_c, c0, z0)
spar = sparsity_jacobian(prob_traj)
∇c_vec = zeros(length(spar))
∇c = zeros(prob_traj.num_con, prob_traj.num_var)
eval_constraint_jacobian!(∇c_vec, z0, prob_traj)
for (i,k) in enumerate(spar)
    ∇c[k[1],k[2]] = ∇c_vec[i]
end
@assert norm(vec(∇c) - vec(∇c_fd)) < 1.0e-10
@assert sum(∇c) - sum(∇c_fd) < 1.0e-10
norm(vec(∇c))# -
