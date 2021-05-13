"""
	loop constraints
"""

struct LoopConstraints <: Constraints
	n
	ineq
	idx
	idx_t1
	idx_t2
	perm
	shift
end

function loop_constraints(model, idx, idx_t1, idx_t2;
	perm = Diagonal(ones(model.n)),
	shift = zeros(model.n))
	n = length(idx)
	ineq = (1:0)
	return LoopConstraints(n, ineq, idx, idx_t1, idx_t2, perm, shift)
end

function constraints!(c, Z, con::LoopConstraints, model, idx, h, T)
	n = con.n
	a = view(Z, idx.x[con.idx_t1][con.idx])
	b = view(Z, idx.x[con.idx_t2][con.idx])

	c[1:n] = con.perm[con.idx, con.idx] * b - (a + con.shift)

    return nothing
end

function constraints_jacobian!(∇c, Z, con::LoopConstraints, model, idx, h, T)
	n = con.n
	shift = 0

	r_idx = 1:n
	c_idx = idx.x[con.idx_t2][con.idx]
	len = length(r_idx) * length(c_idx)
	∇c[shift .+ (1:len)] = vec(con.perm[con.idx, con.idx])
	shift += len

	c_idx = idx.x[con.idx_t1][con.idx]
	len = length(r_idx) * length(c_idx)
	∇c[shift .+ (1:len)] = vec(Diagonal(-1.0 * ones(n)))
	shift += len
	return nothing
end

function constraints_sparsity(con::LoopConstraints, model, idx, T;
		shift_row = 0, shift_col = 0)
	row = []
    col = []

	n = con.n

	r_idx = shift_row .+ (1:n)

	c_idx = shift_col .+ idx.x[con.idx_t2][con.idx]
	row_col!(row, col, r_idx, c_idx)

	c_idx = shift_col .+ idx.x[con.idx_t1][con.idx]
	row_col!(row, col, r_idx, c_idx)

    return collect(zip(row, col))
end
