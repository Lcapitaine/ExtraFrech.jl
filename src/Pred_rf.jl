function pred_rf(frf::Array{Float64,3},Pred::Array{Float64,3},X::Array{Float64,3},X_init::Array{Float64,3}, dist)
    ntree::Int = size(frf,3)
    pred::Array{Float64,2}= zeros(size(X,2), ntree)

    p=Progress(ntree);
    update!(p,0)
    jj = Threads.Atomic{Int}(0)
    l = Threads.SpinLock()

    Threads.@threads for i in 1:ntree
        pred[:,i]= pred_tree( frf[:,:,i], Pred[:,:,i], X,X_init, dist)

        Threads.atomic_add!(jj, 1)
        Threads.lock(l)
        update!(p, jj[])
        Threads.unlock(l)
    end

    return mean(pred,dims=2)
end
