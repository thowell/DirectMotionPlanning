function forward_pass!(p_data::PolicyData, m_data::ModelData, J̄; max_iter = 25)
    status = false

    # line search with rollout
    α = 1.0
    iter = 1
    while true
        iter > max_iter && (@error "forward pass failure", break)

        try
            rollout!(p_data, m_data, α = α)
        catch
            @warn "rollout failure"
        end

        J = objective(m_data.obj, m_data.x, m_data.u)

        if J < J̄
            # update nominal
            m_data.x̄ .= deepcopy(m_data.x)
            m_data.ū .= deepcopy(m_data.u)
            J̄ = J
            status = true
            break
        else
            α *= 0.5
            iter += 1
        end
    end

    return J̄, status
end
