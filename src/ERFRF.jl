function ERFRF(Im::Array{Float64,3}, Y::Vector{Float64}, mtry::Int, ntree::Int, ntry::Int, dist)

    dim::Tuple{Int,Int,Int}=size(Im)
    boot::Array{Float64} = zeros(ntree, dim[2])
    trees::Array{Float64,3} = zeros(4,2*dim[2]-1,ntree)
    P::Array{Float64,3} = zeros(2,2*dim[2]-1,ntree)

    ### On va essayer la barre de progression ::
    p=Progress(ntree);
    update!(p,0)
    jj = Threads.Atomic{Int}(0)
    l = Threads.SpinLock()

    Threads.@threads for k in 1:ntree
        trees_temp , pred_temp, boot_temp = ERtmax(Im,Y,mtry,ntry, dim, dist)
        trees[:,1:size(trees_temp,2),k] = trees_temp
        P[:,1:size(pred_temp,2),k]= pred_temp
        boot[k,1:length(boot_temp)]= boot_temp
        ## on affiche la barre de progression ::
        Threads.atomic_add!(jj, 1)
        Threads.lock(l)
        update!(p, jj[])
        Threads.unlock(l)
    end

    return trees, P, boot
end
