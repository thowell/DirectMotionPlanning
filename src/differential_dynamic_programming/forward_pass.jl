function forward_pass!(p_data::PolicyData, m_data::ModelData, s_data::SolverData;
    max_iter = 25)

    # reset solver status
    s_data.status = false

    # gradient of Lagrangian
    lagrangian_gradient!(s_data, p_data, m_data)
    
    # line search with rollout
    α = 1.0
    iter = 1
    while true
        iter > max_iter && (@error "forward pass failure", break)

        J = Inf
        try
            rollout!(p_data, m_data, α = α)
            J = objective(m_data.obj, m_data.x, m_data.u)
            Δz!(m_data)
        catch
            @warn "rollout failure"
            fill!(m_data.z, 0.0)
        end

        if J < s_data.obj + 0.001 * α * s_data.gradient' * m_data.z
            # update nominal
            m_data.x̄ .= deepcopy(m_data.x)
            m_data.ū .= deepcopy(m_data.u)
            s_data.obj = J
            s_data.status = true
            break
        else
            α *= 0.5
            iter += 1
        end
    end
end
