
function FRFERR(frf::Array{Float64,3},X::Array{Float64,3}, Y::Vector{Float64},P::Array{Float64,3}, id::Array{Float64,2}, dist)
    dim=size(X)::Tuple{Int,Int,Int}
    #variables::Int = dim[3]
    ntree::Int=size(frf,3)
    err_OOB::Vector{Float64}=zeros(ntree)
    pred_OOB::Vector{Float64}=zeros(dim[2])
    ZZ::Array{Float64,3} = zeros(dim[1],2,dim[3])


    for i in 1:dim[2]
        l=1
        Pred_courante = zeros(2,ntree)
        Threads.@threads for k in 1:ntree
            if length(findall(x->x==i,id[k,:]))==1
                ZZ[:,1,:,], ZZ[:,2,:,] = X[:,i,:], X[:,i,:,]
                Pred_courante[1,l] = 1.0
                Pred_courante[2,l] = pred_tree(frf[:,:,i],P[:,:,i],ZZ,X, dist)[1]
                l=l+1
            end

            ## Il faut maintenant regarder la différence en erreur de prédiction ::
        end
        pred_OOB[i] = mean(Pred_courante[2,findall(x->x>0.0,Pred_courante[1,:])])
    end

    mse = mean((Y.-pred_OOB).^2)

    varex = 1 - mse/impurity(Y)
    return pred_OOB, mse, varex
end
