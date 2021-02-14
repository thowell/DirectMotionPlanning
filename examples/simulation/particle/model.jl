"""
    particle dynamics
    - 3D particle subject to contact forces

    - configuration: q = (x, y, z) ∈ R³
    - impact force (magnitude): n ∈ R₊
    - friction force: b ∈ R²
        - contact force: λ = (b, n) ∈ R² × R₊
        - friction coefficient: μ ∈ R₊

    Discrete Mechanics and Variational Integrators
        pg. 363
"""
struct Particle
    n::Int
    m::Int
    d::Int

    mass # mass
    gravity # gravity
    friction_coeff # friction coefficient

    nq # configuration dimension
    nu # control dimension
end

# mass matrix
function mass_matrix(model)
    m = model.mass

    Diagonal(@SVector [m, m, m])
end

# gravity
function gravity(model, q)
    m = model.mass
    g = model.gravity

    @SVector [0.0, 0.0, m * g]
end

# signed distance function
function signed_distance(model, q)
    q[3]
end

# contact force Jacobian
function jacobian(model, q)
    Diagonal(@SVector ones(model.nq))
end

# dynamics
function dynamics(model, q1, q2, q3, u, λ, h)
    nq = model.nq
    SVector{nq}(mass_matrix(model) * (2.0 * q2 - q1 - q3) / h
        - h * gravity(model, q2)
        + h * jacobian(model, q3)' * (u + λ))
end
