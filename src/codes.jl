using Distances
using Statistics
using StatsBase
using Random
using ProgressMeter
using CalculatedABC
using JLD 

function impurity(Y::Vector{Float64})
    if length(Y)<=1
        return 0
    end
    return var(Y)
end

function impurity(Y::Vector{Int})
    return gini_coeff(Y)
end 


function findall3(f, a::AbstractArray{T, N}) where {T, N}
    j = 1
    b = Vector{Int}(undef, length(a))
    @inbounds for i in eachindex(a)
        @inbounds if f(a[i])
            b[j] = i
            j += 1
        end
    end
    resize!(b, j-1)
    sizehint!(b, length(b))
    return b
end



function impurity_split(Y::Vector{T},sp::AbstractVector{Float64}) where T
    l::Int64=length(Y)
    gauche::Vector{Int64} = findall3(x->x==1,sp)
    droite::Vector{Int64} = findall3(x->x==2,sp)
    if length(gauche)==0 || length(droite)==0
        return Inf
    end 
    return impurity(Y[gauche])*length(gauche)/l + impurity(Y[droite])*length(droite)/l
end

### Maintenant la fonction de découpage aléatoire :::


function ERvar_split(Y::Vector{T},Im::Array{Float64,3}, ntry::Int64, dim::Tuple{Int,Int,Int}, dist) where T

    l::Int64=dim[2]

    if l==2

        D::Vector{Float64} = zeros(dim[3])
        @inbounds for p in 1:dim[3]
            D[p]= @views mean(sqrt.((Im[:,1,p]-Im[:,2,p]).^2))
        end
        return findmax(D)[2], [1, 2], [1 ,2], 0

    else
        impur = fill(Inf,dim[3])
        Splits_prime = fill(1,dim[3],dim[2])
        centers = fill(0,2, dim[3])
        whichsplit = 0
        n_dec = min(ntry,l)
        centers_prime = fill(0,2,n_dec)

        for p in 1:dim[3]

            ## plus de tests que d'images ??
            splits = ones(n_dec, l)
            impur_prime = fill(Inf, n_dec)

            for k in 1:n_dec
                centers_prime[:,k] = sample(1:dim[2], 2, replace=false)
                d_g = @views Distances.colwise(dist, Im[:,:,p], Im[:,centers_prime[1,k],p]) ## distance à gauche
                d_d = @views Distances.colwise(dist, Im[:,:,p], Im[:,centers_prime[2,k],p]) ## le fameux dd
                qui = findall3(x->x>=0, d_g-d_d)::Vector{Int}
                splits[k,qui].=  2
                impur_prime[k]= @views impurity_split(Y,splits[k,:])::Float64
            end

            impur[p], whichsplit = findmin(impur_prime)
            Splits_prime[p,:] = @view splits[whichsplit,:]
            centers[:,p]= @view centers_prime[:,whichsplit]
        end

        Imp, whichsplit = findmin(impur)
        return whichsplit, centers[:,whichsplit], Splits_prime[whichsplit,:], Imp
    end

end

### Maintenant on construit les arbres maximaux ::

function ERtmax(Im::Array{Float64,3},Y::Vector{Float64},mtry::Int64,ntry::Int64, minElem::Int,dim::Tuple{Int,Int,Int}, dist)
     ## On va éviter de calculer ça tout le temps !!!! A optimiser !!
    p::Int=dim[3]
    n::Int64= dim[2]
    boot::Vector{Int64} = unique(sample(1:n,n))
    V_split::Array{Float64,2} = zeros(4,length(boot)-1)
    Pred::Array{Float64,2}= ones(2,2*length(boot)-1)


    ### Il faut ensuite construire les données d'apprentissage :
    X_boot::Array{Float64,3} = @view Im[:,boot,:]

    feuilles::Vector{Float64} = ones(length(boot))
    feuilles_prime::Vector{Float64} = feuilles
    ### On passe ensuite à la boucle ::
    decoupe::Int=0
    courant::Int=1
    courant_pred::Int=2

    Pred[2,1]= mean(Y)

    @inbounds for i in 1:length(boot)-1
        ## On va tirer un sous-échantillon de variables possibles ::
        feuilles_prime = feuilles
        if decoupe == length(unique(feuilles))
            break

        else
            @inbounds for j in unique(feuilles)

                qui= collect(findall3(x->x==j, feuilles))
                V::Vector{Int}= sort(sample(1:p, mtry, replace=false))

                ### Maintenant il faut se restreindre aux éléments dans la feuilles
                if length(qui)>minElem
                    Split = ERvar_split(Y[boot][qui],X_boot[:,qui,V], ntry,(dim[1],length(qui),length(V)), dist)
                    gauche::Vector{Int}=findall3(x->x==1, Split[3])
                    droite::Vector{Int}= findall3(x->x==2, Split[3])

                    if length(gauche)>0 && length(droite)>0
                        V_split[1,courant] = j
                        V_split[2,courant] = V[Split[1]]
                        V_split[3,courant] = boot[qui[Split[2][1]]]
                        V_split[4,courant] = boot[qui[Split[2][2]]]


                        feuilles_prime[qui[gauche]].= 2*j#*ones(length(gauche))
                        feuilles_prime[qui[droite]].= (2*j + 1)#*ones(length(droite))

                        Pred[1,courant_pred]=2*j
                        Pred[1,courant_pred+1]=2*j+1

                        Pred[2,courant_pred]= mean(@view Y[boot][qui[gauche]])
                        Pred[2,courant_pred+1] = mean(@view Y[boot][qui[droite]])

                        courant += 1
                        courant_pred +=2
                    else
                        decoupe +=1
                    end
                    

                else
                    decoupe = decoupe+1
                end
            end

            if decoupe < length(unique(feuilles))
                decoupe=0
            end
            feuilles = feuilles_prime
        end
    end
    return V_split[:,1:courant-1], Pred[:,1:courant_pred-1], boot
end


function ERtmax(Im::Array{Float64,3},Y::Vector{Int64},mtry::Int64,ntry::Int64, minElem::Int,dim::Tuple{Int,Int,Int}, dist)
    ## On va éviter de calculer ça tout le temps !!!! A optimiser !!
   p=dim[3]
   n= dim[2]
   boot = unique(sample(1:n,n))
   V_split = zeros(4,length(boot)-1)
   Pred= ones(Int64, 2,2*length(boot)-1)


   ### Il faut ensuite construire les données d'apprentissage :
   X_boot = @view Im[:,boot,:]

   feuilles = ones(length(boot))
   feuilles_prime = feuilles
   ### On passe ensuite à la boucle ::
   decoupe=0
   courant=1
   courant_pred=2

   Pred[2,1]= findmax(countmap(Y))[2]

   @inbounds for i in 1:length(boot)-1
       ## On va tirer un sous-échantillon de variables possibles ::
       feuilles_prime = feuilles
       if decoupe == length(unique(feuilles))
           break

       else
           @inbounds for j in unique(feuilles)

               qui= collect(findall3(x->x==j, feuilles))

               impur = gini_coeff(Y[boot][qui])

               if impur > 0

                    V= sort(sample(1:p, mtry, replace=false))
                    if length(qui)>minElem
                        Split = ERvar_split(Y[boot][qui],X_boot[:,qui,V], ntry,(dim[1],length(qui),length(V)), dist)
                        gauche::Vector{Int}=findall3(x->x==1, Split[3])
                        droite::Vector{Int}= findall3(x->x==2, Split[3])
                        if length(gauche)>0 && length(droite)>0
                            V_split[1,courant] = j
                            V_split[2,courant] = V[Split[1]]
                            V_split[3,courant] = boot[qui[Split[2][1]]]
                            V_split[4,courant] = boot[qui[Split[2][2]]]
                            feuilles_prime[qui[gauche]].= 2*j#*ones(length(gauche))
                            feuilles_prime[qui[droite]].= (2*j + 1)#*ones(length(droite))

                            Pred[1,courant_pred]=2*j
                            Pred[1,courant_pred+1]=2*j+1

                            Pred[2,courant_pred]= findmax(countmap(@view Y[boot][qui[gauche]]))[2]
                            Pred[2,courant_pred+1] = findmax(countmap(@view Y[boot][qui[droite]]))[2]

                            courant += 1
                            courant_pred +=2
                        else
                            decoupe +=1
                        end
                    else
                        decoupe = decoupe+1
                    end
                else 
                    decoupe += 1 
                end 
            end 

            if decoupe < length(unique(feuilles))
                decoupe=0
            end

           feuilles = feuilles_prime
       end
   end
   return V_split[:,1:courant-1], Pred[:,1:courant_pred-1], boot
end


### On va passer aux forêts aléatoires ::::

function ERFRF(Im::Array{Float64,3}, Y::Vector{Float64}, mtry::Int, ntree::Int, ntry::Int, minElem::Int, dist) 
    ### Allez c'est parti !!
    ### on transforme les images en données spatio-Temporelles::
    #### On passe à la suite :::

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
        trees_temp , pred_temp, boot_temp = ERtmax(Im,Y,mtry,ntry,minElem, dim, dist)
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

function ERFRF(Im::Array{Float64,3}, Y::Vector{Int64}, mtry::Int, ntree::Int, ntry::Int, minElem::Int, dist) 
    ### Allez c'est parti !!
    ### on transforme les images en données spatio-Temporelles::
    #### On passe à la suite :::

    dim::Tuple{Int,Int,Int}=size(Im)
    boot::Array{Float64} = zeros(ntree, dim[2])
    trees::Array{Float64,3} = zeros(4,2*dim[2]-1,ntree)
    P::Array{Int64,3} = zeros(Int64,2,2*dim[2]-1,ntree)

    ### On va essayer la barre de progression ::
    p=Progress(ntree);
    update!(p,0)
    jj = Threads.Atomic{Int}(0)
    l = Threads.SpinLock()

    Threads.@threads for k in 1:ntree
        trees_temp , pred_temp, boot_temp = ERtmax(Im,Y,mtry,ntry,minElem, dim, dist)
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

### On passe à la fonction de prédiction sur les arbres :::

function pred_tree(tree::AbstractArray{Float64,2},Pred::AbstractArray{T,2},X::AbstractArray{Float64,3},X_init::AbstractArray{Float64,3}, dist) where T

    pred::Vector{T} = zeros(T,size(X,2))
    nodes::Vector{Int64} = ones(Int64,size(X,2))
    feuilles::Vector{Int64}= zeros(Int64,size(X,2))

    #tree = convert(Array{Int64,3},tree) ## A tester pour voir si on peut pas couper dans le gras ?

    while sum(feuilles)<size(X,2)
        @inbounds for i in unique(nodes)
            qui::Vector{Int64} = findall3(x->x==i,nodes)
            col::Vector{Int64} = findall3(x->x==i,@view tree[1,:])
            if length(col)>0
                d_g::Vector{Float64}= @views Distances.colwise(dist, X[:,qui,convert(Int64,tree[2,col][1])], X_init[:,convert(Int64,tree[3,col][1]),convert(Int64,tree[2,col][1])])
                d_d::Vector{Float64}= @views Distances.colwise(dist, X[:,qui,convert(Int64,tree[2,col][1])], X_init[:,convert(Int64,tree[4,col][1]),convert(Int64,tree[2,col][1])])
                gauche::Vector{Int64} = findall3(x->x<=0, d_g-d_d)
                droite::Vector{Int64} = findall3(x->x>0, d_g-d_d)
                nodes[qui[gauche]] = 2*nodes[qui[gauche]]
                nodes[qui[droite]] = 2*nodes[qui[droite]] .+ 1

            elseif unique(feuilles[qui])[1]==0.0
                pred[qui] .=  @view Pred[2,findall3(x->x==i,@view Pred[1,:])]
                feuilles[qui] .= 1
            end

        end
    end
    return pred
end



function Importance(frf::Array{Float64,3},X::Array{Float64,3}, Y::Vector{Float64},P::Array{Float64,3}, id::Array{Float64,2}, dist)
    dim=size(X)::Tuple{Int,Int,Int}
    variables::Int = dim[3]
    imp::Vector{Float64}= zeros(dim[3])
    ntree::Int=size(frf,3)
    err_courante::Vector{Float64}=zeros(ntree)

    prog =Progress(variables);
    update!(prog ,0)

    ID::Array{Int64,2} = convert(Array{Int64,2},id)

    @inbounds for p in 1:variables
        err_courante.=0
        Threads.@threads for i in 1:ntree
            boot::Vector{Int64}= ID[i,findall3(x->x>0.0,ID[i,:])]
            ### Trouver les éléments qui ne sont pas dans l'échantillon OOB :
            OOB::Vector{Int64} = setdiff(1:dim[2],boot)
            X_permute::Array{Float64,3} =  X[:,OOB,:]
            X_permute[:,:,p] = @view X[:,OOB[randperm(length(OOB))],p]
            ## Il faut maintenant regarder la différence en erreur de prédiction ::
            err_courante[i] = @views mean((Y[OOB].-pred_tree(frf[:,:,i], P[:,:,i],X_permute,X, dist)).^2)-mean((Y[OOB].-pred_tree(frf[:,:,i],P[:,:,i],X[:,OOB,:],X, dist)).^2)
        end
        imp[p]= mean(err_courante)
        update!(prog, p)
    end
    return imp
end


function Importance(frf::Array{Float64,3},X::Array{Float64,3}, Y::Vector{Int64},P::Array{Int64,3}, id::Array{Float64,2}, dist)
    dim=size(X)::Tuple{Int,Int,Int}
    variables::Int = dim[3]
    imp::Vector{Float64}= zeros(dim[3])
    ntree::Int=size(frf,3)
    err_courante::Vector{Float64}=zeros(ntree)

    prog =Progress(variables);
    update!(prog ,0)

    ID::Array{Int64,2} = convert(Array{Int64,2},id)

    @inbounds for p in 1:variables
        err_courante.=0
        Threads.@threads for i in 1:ntree
            boot::Vector{Int64}= ID[i,findall3(x->x>0.0,ID[i,:])]
            ### Trouver les éléments qui ne sont pas dans l'échantillon OOB :
            OOB::Vector{Int64} = setdiff(1:dim[2],boot)
            X_permute::Array{Float64,3} =  X[:,OOB,:]
            X_permute[:,:,p] = @view X[:,OOB[randperm(length(OOB))],p]
            ## Il faut maintenant regarder la différence en erreur de prédiction ::
            err_courante[i] = @views mean((Y[OOB].!=pred_tree(frf[:,:,i], P[:,:,i],X_permute,X, dist)))-mean((Y[OOB].!=pred_tree(frf[:,:,i],P[:,:,i],X[:,OOB,:],X, dist)))
        end
        imp[p]= mean(err_courante)
        update!(prog, p)
    end
    return imp
end


### On fait la fonction du calcul du ‰ de variance expliquée ::
### On va aussi rajouter les prediction OOB pour la forêt complète ::

function FRFERR(frf::Array{Float64,3},X::Array{Float64,3}, Y::Vector{Float64},P::Array{Float64,3}, id::Array{Float64,2}, dist)
    dim=size(X)::Tuple{Int,Int,Int}
    #variables::Int = dim[3]
    ntree::Int=size(frf,3)

    pred_OOB::Vector{Float64}=zeros(dim[2])
    ZZ::Array{Float64,3} = zeros(dim[1],2,dim[3])


    @inbounds for i in 1:dim[2]
        Pred_courante = fill(Inf,2,ntree)
        ZZ[:,1,:], ZZ[:,2,:] = X[:,i,:], X[:,i,:]
        Threads.@threads for k in 1:ntree
            if length(findall3(x->x==i,id[k,:]))==0
                Pred_courante[1,k] = 1.0
                Pred_courante[2,k] = @views pred_tree(frf[:,:,k],P[:,:,k],ZZ,X, dist)[1]
            end

            ## Il faut maintenant regarder la différence en erreur de prédiction ::
        end
        pred_OOB[i] = mean(@views Pred_courante[2,findall3(x->x<Inf,Pred_courante[1,:])])
    end

    mse = mean((Y.-pred_OOB).^2)

    varex = 1 - (mse/impurity(Y))
    return pred_OOB, mse, varex
end

function FRFERR(frf::Array{Float64,3},X::Array{Float64,3}, Y::Vector{Int64},P::Array{Int64,3}, id::Array{Float64,2}, dist)
    dim=size(X)
    #variables::Int = dim[3]
    ntree=size(frf,3)

    pred_OOB=zeros(Int64,dim[2])
    ZZ = zeros(dim[1],2,dim[3])


    @inbounds for i in 1:dim[2]
        Pred_courante = fill(-1,2,ntree)
        ZZ[:,1,:], ZZ[:,2,:] = X[:,i,:], X[:,i,:]
        Threads.@threads for k in 1:ntree
            if length(findall3(x->x==i,id[k,:]))==0
                Pred_courante[1,k] = 0
                Pred_courante[2,k] = @views pred_tree(frf[:,:,k],P[:,:,k],ZZ,X, dist)[1]
            end

            ## Il faut maintenant regarder la différence en erreur de prédiction ::
        end
        pred_OOB[i] = findmax(countmap(@views Pred_courante[2,findall3(x->x==0,Pred_courante[1,:])]))[2]
    end

    mse = mean(Y.!=pred_OOB)

    init_imp = mean(Y .!= findmax(countmap(Y))[2])

    varex = 1 - (mse/init_imp)
    return pred_OOB, mse, varex
end


### On regarde maintenant les predictions OOB de la forêt de Fréchet pour calculer le % de var expliquée:::

function Importance_Unique(frf::Array{Char},X::Array{Float64,3}, Y::Vector{Float64},variables::UnitRange{Int64}, dist)
    dim=size(X)::Tuple{Int,Int,Int}
    imp::Array{Float64,2}= zeros(2,length(variables))

    trees=readdir(frf; join = true)
    ntree::Int=length(trees)

    err_courante::Vector{Float64}=zeros(ntree)
    imp[1,:] = variables
    ID::Array{Int64,2} = convert(Array{Int64,2},id)


    @inbounds for p in 1:length(variables)
        err_courante.=0
        Threads.@threads for i in 1:ntree

            infos = load(trees[k])
            tree = infos["tree"]
            ID = infos["boot"]
            P = infos["P"]

            boot::Vector{Int64}= @view ID[i,findall3(x->x>0.0,ID[i,:])]
            ### Trouver les éléments qui ne sont pas dans l'échantillon OOB :
            OOB::Vector{Int64} = setdiff(1:dim[2],boot)


            X_permute::Array{Float64,3} = @views X[:,OOB,:]
            X_permute[:,:,variables[p]] = @views X[:,OOB[randperm(length(OOB))],variables[p]]

            ## Il faut maintenant regarder la différence en erreur de prédiction ::
            err_courante[i] = @views mean((Y[OOB].-pred_tree(tree, P,X_permute,X, dist)).^2)-mean((Y[OOB].-pred_tree(tree,P,X[:,OOB,:],X, dist)).^2)
        end
        imp[2,p]= mean(err_courante)
    end
    return imp
end

function Importance_Unique(frf::Array{Char},X::Array{Float64,3}, Y::Vector{Int64}, variables::UnitRange{Int64}, dist)
    dim=size(X)::Tuple{Int,Int,Int}
    imp::Array{Float64,2}= zeros(2,length(variables))

    trees=readdir(frf; join = true)
    ntree::Int=length(trees)

    err_courante::Vector{Float64}=zeros(ntree)
    imp[1,:] = variables

    @inbounds for p in 1:length(variables)
        err_courante.=0
        Threads.@threads for i in 1:ntree

            infos = load(trees[k])
            tree = infos["tree"]
            ID = infos["boot"]
            P = infos["P"]

            boot::Vector{Int64}= @view ID[i,findall3(x->x>0.0,ID[i,:])]
            ### Trouver les éléments qui ne sont pas dans l'échantillon OOB :
            OOB::Vector{Int64} = setdiff(1:dim[2],boot)
            X_permute::Array{Float64,3} = @views X[:,OOB,:]
            X_permute[:,:,variables[p]] = @views X[:,OOB[randperm(length(OOB))],variables[p]]
            ## Il faut maintenant regarder la différence en erreur de prédiction ::
            err_courante[i] = @views mean((Y[OOB].!=pred_tree(tree, P,X_permute,X, dist)))-mean((Y[OOB].!=pred_tree(tree,P,X[:,OOB,:],X, dist)))
        end
        imp[2,p]= mean(err_courante)
    end
    return imp
end

function ExtraFrechetRF(X::Array{Float64,3}, Y::Vector{Float64}, mtry::Int, ntree::Int, ntry::Int, imp::Bool,minElem::Int, dist)
    println("Building the Extra Fréchet Forest:")
    frf, P, boot  = ERFRF(X,Y, mtry, ntree, ntry,minElem, dist)
    println("OOB errors and % of explained variance:")
    pred_OOB, mse, varex = FRFERR(frf,X, Y,P, boot, dist)

    if imp==true
        println("Variables Importance Scores:")
        Imp = Importance(frf,X,Y,P, boot, dist)
        return Dict("FrechetRF"=> frf, "Imp"=>Imp, "varex"=> varex, "mse"=>mse, "OOB pred"=>pred_OOB, "P"=>P, "mtry"=>mtry, "ntry"=>ntry, "ntree"=>ntree, "dist"=>dist)
    end 

    return Dict("FrechetRF"=> frf, "Imp"=>false, "varex"=> varex, "mse"=>mse, "OOB pred"=>pred_OOB, "P"=>P, "mtry"=>mtry, "ntry"=>ntry, "ntree"=>ntree, "dist"=>dist)
end


function ExtraFrechetRF(X::Array{Float64,3}, Y::Vector{Int64}, mtry::Int, ntree::Int, ntry::Int, imp::Bool,minElem::Int, dist)
    println("Building the Extra Fréchet Forest:")
    frf, P, boot  = ERFRF(X,Y, mtry, ntree, ntry,minElem, dist)
    println("OOB errors and % of explained variance:")
    pred_OOB, mse, varex = FRFERR(frf,X, Y,P, boot, dist)

    if imp==true
        println("Variables Importance Scores:")
        Imp = Importance(frf,X,Y,P, boot, dist)
        return Dict("FrechetRF"=> frf, "Imp"=>Imp, "varex"=> varex, "Class error"=>mse, "OOB pred"=>pred_OOB, "P"=>P, "mtry"=>mtry, "ntry"=>ntry, "ntree"=>ntree, "dist"=>dist)
    end 

    return Dict("FrechetRF"=> frf, "Imp"=>false, "varex"=> varex, "Class error"=>mse, "OOB pred"=>pred_OOB, "P"=>P, "mtry"=>mtry, "ntry"=>ntry, "ntree"=>ntree, "dist"=>dist)
end


function prediction_rf(pred::Array{Float64,2})
    return mean(pred,dims=2)
end 

function prediction_rf(pred::Array{Int64,2})

    sortie = fill(0, size(pred,2))
    for i in 1:size(pred,2)
        sortie[i] = findmax(countmap(pred[:,i]))[2]
    end 
    return sortie
end 


function pred_rf(frf::Dict{String, Any}, X::Array{Float64,3}, X_init::Array{Float64,3})
    ntree::Int = get(frf, "ntree",1)
    pred::Array{Float64,2}= zeros(size(X,2), ntree)

    forest = get(frf, "FrechetRF", 1)
    Pred = get(frf, "P", 1)
    dist = get(frf, "dist",1)

    p=Progress(ntree);
    update!(p,0)
    jj = Threads.Atomic{Int}(0)
    l = Threads.SpinLock()

    Threads.@threads for i in 1:ntree
        pred[:,i]= pred_tree(forest[:,:,i], Pred[:,:,i], X,X_init, dist)

        Threads.atomic_add!(jj, 1)
        Threads.lock(l)
        update!(p, jj[])
        Threads.unlock(l)
    end

    return prediction_rf(pred)
end



function OOB_unique(frf::Dict{String, Array}, X::Array{Float64,3}, indiv::Vector{Int64}, dist)

    dim=size(X)

    ntree = size(frf["trees"],3)
    type = eltype(frf["P"][1,1,1])

    pred_OOB=zeros(type,length(indiv))
    proba = zeros(length(indiv))
    ZZ = zeros(dim[1],2,dim[3])


    @inbounds for i in 1:length(indiv)
        Pred_courante = fill(-1,2,ntree)
        ZZ[:,1,:], ZZ[:,2,:] = X[:,indiv[i],:], X[:,indiv[i],:]
        Threads.@threads for k in 1:ntree

            if length(findall3(x->x==indiv[i],frf["id"][k,:])) == 0
                Pred_courante[1,k] = 0
                Pred_courante[2,k] = @views pred_tree(frf["trees"][:,:,k],frf["P"][:,:,k],ZZ,X, dist)[1]
            end

            ## Il faut maintenant regarder la différence en erreur de prédiction ::
        end

        if type==Int
            
            elem  = findmax(countmap(@views Pred_courante[2,findall3(x->x==0,Pred_courante[1,:])]))
            pred_OOB[i] = elem[2]
            proba[i] = elem[1]/length(findall3(x->x==0,Pred_courante[1,:]))
        else 
            pred_OOB[i] = mean(@views Pred_courante[2,findall3(x->x==0,Pred_courante[1,:])])
        end 
    end
    return pred_OOB, proba 
end


function Combine(lien::String,X::Array{Float64,3})

    dim = size(X)

    Trees=readdir(lien, join = true)
    ntree=length(Trees)

    type = eltype(load(Trees[1])["P"][1,1])

    boot = zeros(ntree, dim[2])
    trees = zeros(4,2*dim[2]-1,ntree)
    P = zeros(type,(2,2*dim[2]-1,ntree))

    for k in 1:ntree

        infos = load(Trees[k])
        tree_temp = infos["tree"]
        id_temp = infos["boot"]
        P_temp = infos["P"]

        trees[:,1:size(tree_temp,2),k] = tree_temp
        P[:,1:size(P_temp,2),k]= P_temp
        boot[k,1:length(id_temp)]= id_temp

    end 

    return Dict("trees"=>trees, "P"=>P, "id"=>boot)
end 

function Combine_OOB(racine::String,Y)

    cd(string(racine,"/OOB"))
    files = readdir(pwd(), join=true)
    pred_OOB = []
    proba_OOB = []
    var_init = mean(Y.!=findmax(countmap(Y))[2])

    for i in 1:length(files)
        infos = load(string("OOB_",i,".jld"))
        pred_OOB = vcat(pred_OOB,infos["pred"])
        proba_OOB = vcat(proba_OOB, infos["proba"])
    end 

    mse = mean(Y.!= pred_OOB)
    varex = 1 - mse/var_init

    return pred_OOB, proba_OOB, mse, varex 
end 