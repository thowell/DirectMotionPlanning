function forward_pass!(p_data::PolicyData, m_data::ModelData, s_data::SolverData;
    max_iter = 25)

    # reset solver status
    s_data.status = false

    # previous cost
    J_prev = s_data.obj

    # gradient of Lagrangian
    lagrangian_gradient!(s_data, p_data, m_data)

    # line search with rollout
    s_data.α = 1.0
    iter = 1
    while true
        iter > max_iter && (@error "forward pass failure", break)

        J = Inf
        try
            rollout!(p_data, m_data, α = s_data.α)
            J = objective!(s_data, m_data, mode = :current)
            Δz!(m_data)
        catch
            @warn "rollout failure"
            fill!(m_data.z, 0.0)
        end

        if J < J_prev + 0.001 * s_data.α * s_data.gradient' * m_data.z
            # update nominal
            m_data.x̄ .= deepcopy(m_data.x)
            m_data.ū .= deepcopy(m_data.u)
            s_data.obj = J
            s_data.status = true
            break
        else
            s_data.α *= 0.5
            iter += 1
        end
    end
end
