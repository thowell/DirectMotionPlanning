# test differentiable MPC policy on cartpole swingup
include_model("cartpole")
n = model.n
m = model.m
d = model.d

# fast dynamics
@variables x1[1:n], x2[1:n], u1[1:m], w1[1:d], hs[1:1], ts[1:1]

function fd(model::Cartpole{Midpoint, FixedTime}, x⁺, x, u, w, h, t)
    x⁺ - (x + h[1] * f(model, 0.5 * (x + x⁺), u, w))
end

_dynamics = fd(model, x2, x1, u1, w1, hs, ts)
dynamics = eval(build_function(_dynamics, x2, x1, u1, w1, hs, ts)[2])
∇dynamics_x2 = eval(build_function(Symbolics.jacobian(_dynamics, x2), x2, x1, u1, w1, hs, ts)[1])
∇dynamics_x1 = eval(build_function(Symbolics.jacobian(_dynamics, x1), x2, x1, u1, w1, hs, ts)[1])
∇dynamics_u1 = eval(build_function(Symbolics.jacobian(_dynamics, u1), x2, x1, u1, w1, hs, ts)[1])

# fast objective
function objective(x, u, t)
    J = 0.0
    if t < 10
        J += transpose(x) * x
        J += transpose(u) * u
    else
        J += 10.0 * transpose(x) * x
    end
    return J
end

_obj = objective(x1, u1, t)
obj = eval(build_function)
function objective(x)
    J = 0.0
    J += transpose(x) * x
    return J
end
