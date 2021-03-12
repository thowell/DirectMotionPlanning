using ModelingToolkit

@variables a
function huber(a)
    δ = 1.0
    if abs(a) <= δ
        return 0.5*a^2
    else
        return δ*(abs(a) - 0.5*δ)
    end
end
da = huber(a)
