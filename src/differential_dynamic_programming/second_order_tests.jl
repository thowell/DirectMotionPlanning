using Symbolics

include_model("quadrotor")

function fd(model::Quadrotor{Midpoint, FixedTime}, x, u, w, h, t)
    x + h[1] * f(model, x + 0.5 * h[1] * f(model, x, u, w), u, w)
end

n = model.n
m = model.m
d = model.d

@variables x_sym[1:n]
@variables u_sym[1:m]
@variables w_sym[1:d]
@variables h_sym[1:1]
@variables t_sym[1:1]
@variables P_sym[1:n, 1:n]
@variables p_sym[1:n]

fd_sym = fd(model, x_sym, u_sym, w_sym, h_sym, t_sym)

Q = 10.0 * Diagonal(ones(n))
R = 1.0 * Diagonal(ones(m))

function gt(x, u, t)
    transpose(x) * Q * x + transpose(u) * R * u
end

gt_sym = gt(x_sym, u_sym, t_sym)

function Vt(x, P, p)
    0.5 * transpose(x) * P * x + transpose(x) * p
end

V_sym = Vt(x_sym, P_sym, p_sym)

function Qt(x, u, P⁺, p⁺, h, t)
    x⁺ = fd(model, x, u, zeros(model.d), h, t)
    gt(x, u, t) + Vt(x⁺, P⁺, p⁺)
end

Qt_sym = Qt(x_sym, u_sym, P_sym, p_sym, h_sym, t_sym)

Qt_grad = Symbolics.gradient(Qt_sym, [x_sym; u_sym], simplify = true)
Qt_hess = Symbolics.hessian(Qt_sym, [x_sym; u_sym], simplify = true)
Qt_hess_sp = Symbolics.sparsehessian(Qt_sym, [x_sym; u_sym], simplify = true)
