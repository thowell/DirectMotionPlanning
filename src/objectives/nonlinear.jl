"""
    nonlinear objective
"""
struct NonlinearObjective <: Objective
    f # objectivce
    g # objective gradient
    gradient_type
end

function nonlinear_objective(f)
    NonlinearObjective(f, x -> zero(x), :auto)
end

function nonlinear_objective(f, g)
    NonlinearObjective(f, g, :gradient)
end

function objective(Z, obj::NonlinearObjective, idx, T)
    return obj.f(Z)
end

function objective_gradient!(∇J, Z, obj::NonlinearObjective, idx, T)
    if obj.gradient_type == :auto
        ∇J .+= ForwardDiff.gradient(obj.f, Z)
    else
        ∇J .+= obj.g(Z)
    end
    return nothing
end
