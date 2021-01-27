"""
    finite-difference control velocity objective
"""
struct ControlVelocityObjective <: Objective
    R
end

function control_velocity_objective(R)
    ControlVelocityObjective(R)
end

function objective(Z, obj::ControlVelocityObjective, idx, T)
    J = 0.0

    for t = 1:T-2
        u⁻ = view(Z, idx.u[t])
        u⁺ = view(Z, idx.u[t+1])
        cv = (u⁺ - u⁻)
        J += cv' * obj.R * cv
    end

    return J
end

function objective_gradient!(∇J, Z, obj::ControlVelocityObjective, idx, T)
    J = 0.0
    for t = 1:T-2
        u⁻ = view(Z, idx.u[t])
        u⁺ = view(Z, idx.u[t+1])
        cv = (u⁺ - u⁻)
        # J += cv' * obj.R * cv

        dJdv = 2.0 * obj.R * cv
        ∇J[idx.u[t]] -= dJdv
        ∇J[idx.u[t + 1]] += dJdv
    end

    return nothing
end
