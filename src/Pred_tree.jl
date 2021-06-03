function pred_tree(tree::Array{Float64,2},Pred::Array{Float64,2},X::Array{Float64,3},X_init::Array{Float64,3}, dist)

    pred::Vector{Float64} = zeros(size(X,2))
    nodes::Vector{Float64} = ones(size(X,2))
    feuilles::Vector{Float64}= zeros(size(X,2))

    #tree = convert(Array{Int64,3},tree) ## A tester pour voir si on peut pas couper dans le gras ?

    while sum(feuilles)<size(X,2)
        for i in unique(nodes)
            qui::Vector{Int64} = findall(x->x==i,nodes)
            col::Vector{Int64} = findall(x->x==i,@view tree[1,:])
            if length(col)>0
                d_g::Vector{Float64}= Distances.colwise(dist, X[:,qui,convert(Int64,tree[2,col][1])], X_init[:,convert(Int64,tree[3,col][1]),convert(Int64,tree[2,col][1])])
                d_d::Vector{Float64}= Distances.colwise(dist, X[:,qui,convert(Int64,tree[2,col][1])], X_init[:,convert(Int64,tree[4,col][1]),convert(Int64,tree[2,col][1])])
                gauche::Vector{Int64} = findall(x->x<=0, d_g-d_d)
                droite::Vector{Int64} = findall(x->x>0, d_g-d_d)
                nodes[qui[gauche]] = 2*nodes[qui[gauche]]
                nodes[qui[droite]] = 2*nodes[qui[droite]] .+ 1

            elseif unique(feuilles[qui])[1]==0.0
                pred[qui] .=  @view Pred[2,findall(x->x==i,@view Pred[1,:])]
                feuilles[qui] .= 1
            end
        end
    end
    return pred
end
