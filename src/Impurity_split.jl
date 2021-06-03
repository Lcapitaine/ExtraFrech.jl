function impurity_split(Y::Vector{Float64},sp::Vector{Float64})
    l::Int=length(Y)
    gauche::Vector{Int} = findall(x->x==1,sp)
    droite::Vector{Int} = findall(x->x==2,sp)
    return impurity(Y[gauche])*length(gauche)/l + impurity(Y[droite])*length(droite)/l
end
