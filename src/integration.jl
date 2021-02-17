"""
    continuous model
"""
abstract type Continuous <: Integration end

"""
    explicit midpoint integrator
"""
struct Midpoint <: Integration end

function fd(model::Model{Midpoint, FixedTime}, x, u, w, h, t)
    x + h * f(model, x + 0.5 * h * f(model, x, u, w), u, w)
end

function fd(model::Model{Midpoint, FreeTime}, x, u, w, h, t)
    h = u[end]
    x + h * f(model, x + 0.5 * h * f(model, x, u, w), u, w)
end

"""
    implicit midpoint integrator
"""
function fd(model::Model{Midpoint, FixedTime}, x⁺, x, u, w, h, t)
    x⁺ - (x + h * f(model, 0.5 * (x + x⁺), u, w))
end

function fd(model::Model{Midpoint, FreeTime}, x⁺, x, u, w, h, t)
    h = u[end]
    x⁺ - (x + h * f(model, 0.5 * (x + x⁺), u, w))
end

"""
    explicit RK3 integrator
"""
abstract type RK3 <: Integration end

function fd(model::Model{RK3, FixedTime}, z, u, w, h, t)
    # k1 = k2 = k3 = zero(z)
    k1 = h * f(model, z, u, w)
    k2 = h * f(model, z + 0.5 * k1, u, w)
    k3 = h * f(model, z - k1 + 2.0 * k2, u, w)
    z + (k1 + 4.0 * k2 + k3) / 6.0
end

abstract type RK4 <: Integration end

function fd(model::Model{RK4, FixedTime}, z, u, w, h, t)
	k1 = f(model, z, u, w) * h
	k2 = f(model, z + 0.5 * k1, u, w) * h
	k3 = f(model, z + 0.5 * k2, u, w) * h
	k4 = f(model, z + k3, u, w) * h
	z + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
end

"""
    discrete explicit dynamics
"""
abstract type Discrete <: Integration end
