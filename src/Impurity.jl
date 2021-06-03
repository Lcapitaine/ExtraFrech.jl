function impurity(Y::Vector{Float64})
    if length(Y)<=1
        return 0
    end
    return var(Y)
end
