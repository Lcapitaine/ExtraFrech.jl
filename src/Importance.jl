function Importance(frf::Array{Float64,3},X::Array{Float64,3}, Y::Vector{Float64},P::Array{Float64,3}, id::Array{Float64,2}, dist)
    dim=size(X)::Tuple{Int,Int,Int}
    variables::Int = dim[3]
    imp::Vector{Float64}= zeros(dim[3])
    ntree::Int=size(frf,3)
    err_courante::Vector{Float64}=zeros(ntree)

    prog =Progress(variables);
    update!(prog ,0)

    ID::Array{Int64,2} = convert(Array{Int64,2},id)

    for p in 1:variables
        err_courante.=0
        Threads.@threads for i in 1:ntree
            boot::Vector{Int64}= ID[i,findall(x->x>0.0,ID[i,:])]
            ### Trouver les éléments qui ne sont pas dans l'échantillon OOB :
            OOB::Vector{Int64} = setdiff(1:dim[2],boot)
            X_permute::Array{Float64,3} =  X[:,OOB,:]
            X_permute[:,:,p] = @view X[:,OOB[randperm(length(OOB))],p]
            ## Il faut maintenant regarder la différence en erreur de prédiction ::
            err_courante[i] = mean((Y[OOB].-pred_tree(frf[:,:,i], P[:,:,i],X_permute,X, dist)).^2)-mean((Y[OOB].-pred_tree(frf[:,:,i],P[:,:,i],X[:,OOB,:],X, dist)).^2)
        end
        imp[p]= mean(err_courante)
        update!(prog, p)
    end
    return imp
end
