using Distances
using Statistics
using StatsBase
using Random
using ProgressMeter
using SIMD
using LoopVectorization


function findall3(f, a::Array{T, N}) where {T, N}
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


function impurity(Y::Vector{Float64})
    if length(Y)<=1
        return 0
    end
    return var(Y)
end


function impurity_split(Y::AbstractVector{Float64},sp::Vector{Int8})
    gauche::Vector{Int64} = findall3(x->x==1,sp)
    droite::Vector{Int64} = findall3(x->x==2,sp)
    if length(gauche)>1
        if length(droite)>1 
            return (Statistics.var(@view Y[gauche])*length(gauche) + Statistics.var(@view Y[droite])*length(droite))/length(Y)
        else 
            return Statistics.var(@view Y[gauche])*length(gauche)/length(Y)
        end 
    elseif length(droite)>1 
        return (Statistics.var(@view Y[droite])*length(droite)/length(Y))
    end 
    return 0 
end


function ERvar_split(Y::AbstractVector{Float64},Im::AbstractArray{Float64,3}, ntry::Int64, dim::Tuple{Int,Int,Int},dist::Any)

    l::Int64=dim[2]

    if l==2
        
        D::Vector{Float64} = zeros(dim[3])
        @inbounds for p in 1:dim[3]
            D[p]= @views mean(sqrt.((Im[:,1,p]-Im[:,2,p]).^2))
        end
        return findmax(D)[2], [1, 2], [1 ,2], 0

    else
        impur::Float64= Inf
        Split::Vector{Int8} = ones(Int8,l)
    
        centers::Vector{Int64} = [1,1]
        whichsplit::Int64 = zero(Int64)
        n_dec::Int64 = min(ntry,l)

        ### Maintenant on va pouvoir tester la fonction globale :: 
        ### Il faut juste copier les différents paramètres courants : 

        impur_courant::Float64 = copy(impur)
        Splits_prime::Vector{Int8} = copy(Split)
        centers_prime::Vector{Int64} = copy(centers)

        @inbounds for p in 1:dim[3]

            @inbounds @simd for k in 1:n_dec
                Splits_prime .= 1
                centers_prime .= sample(1:dim[2], 2, replace=false)
                d_g::Vector{Float64} = @views Distances.colwise(dist,  Im[:,:,p], Im[:,centers_prime[1],p]) ## distance à gauche
                d_d::Vector{Float64} = @views Distances.colwise(dist,  Im[:,:,p], Im[:,centers_prime[2],p]) ## le fameux dd

                qui = findall3(x->x>=0, d_g-d_d)
                Splits_prime[qui] .= 2 
                impur_courant = impurity_split(Y,Splits_prime)

                if impur_courant < impur
                    whichsplit = p 
                    impur = impur_courant
                    Split = Splits_prime
                    centers = centers_prime
                end 

            end
        end
        return whichsplit, centers, Split
    end
end


function ERtmax(Im::Array{Float64,3},Y::Vector{Float64},mtry::Int64,ntry::Int64,minElem::Int64,dist)
    ## On va éviter de calculer ça tout le temps !!!! A optimiser !!
   dim = size(Im)
   p=dim[3]::Int64 
   n= dim[2]::Int64
   boot = unique(sample(1:n,n))::Vector{Int64}

   noeuds = zeros(Int64, length(boot)-1)
   V_split = zeros(Int64, length(boot)-1)
   Centers = zeros(Int64, (length(boot)-1,2))
   Split = zeros(Int8, length(boot))

   V_split_courant = zero(Int64)
   Centers_courant = zeros(Int64,2)
   

   Pred= ones(2,2*length(boot)-1)::Array{Float64,2}
   


   ### Il faut ensuite construire les données d'apprentissage :
   feuilles = ones(Int64, length(boot))::Vector{Int64}
   feuilles_prime = feuilles::Vector{Int64}
   decoupe=0
   courant=1
   courant_pred=2

   Pred[2,1]= mean(Y)

   @inbounds for i in 1:length(boot)/2-1

       feuilles_prime = feuilles
       if decoupe == length(unique(feuilles))
           break

       else
           @inbounds @simd for j in unique(feuilles)

               qui= collect(findall3(x->x==j, feuilles))::Vector{Int}
               V= sort(sample(1:p, mtry, replace=false))::Vector{Int}

               if length(qui)>minElem

                   V_split_courant, Centers_courant, Split[qui] = @views ERvar_split(Y[boot[qui]],X[:,boot[qui],V], ntry,(dim[1],length(qui),length(V)), dist)

                   gauche=findall3(x->x==1, Split[qui])::Vector{Int}
                   droite= findall3(x->x==2, Split[qui])::Vector{Int}
                   

                   if length(gauche)>0 && length(droite)>0

                       V_split[courant] = V_split_courant
                       noeuds[courant] = j
                       Centers[courant,:]= Centers_courant

                       feuilles_prime[qui[gauche]] .= 2*j
                       feuilles_prime[qui[droite]] .= (2*j + 1)

                       Pred[1,courant_pred] = 2*j
                       Pred[1,courant_pred+1] = 2*j+1

                       Pred[2,courant_pred] =  mean(@view Y[boot][qui[gauche]])
                       Pred[2,courant_pred+1] =  mean(@view Y[boot][qui[droite]])

                       courant += 1
                       courant_pred += 2
                   else
                       decoupe += 1
                   end
                   

               else
                   decoupe = decoupe+1
               end
           end

           if decoupe < length(unique(feuilles))
               decoupe = 0
           end
           feuilles = feuilles_prime
       end
   end
   return noeuds[1:courant-1], V_split[1:courant-1], Centers, Pred[:,1:courant_pred-1], boot
end


function ERFRF(Im::Array{Float64,3}, Y::Vector{Float64}, mtry::Int64, ntree::Int64, ntry::Int64,minElem::Int64, dist)
    ### Allez c'est parti !!
    ### on transforme les images en données spatio-Temporelles::
    #### On passe à la suite :::

    dim = size(Im)::Tuple{Int64,Int64,Int64}

    boot = zeros(Int64, (ntree, dim[2]))::Array{Int64}
    nodes = zeros(Int64, (2*dim[2]-1,ntree))
    Vsplit = zeros(Int64, (2*dim[2]-1,ntree))
    centres = zeros(Int64, (2*dim[2]-1,2,ntree))::Array{Int64,3}
    P= zeros(2,2*dim[2]-1,ntree)::Array{Float64,3} 

    ### On va essayer la barre de progression ::
    p=Progress(ntree);
    update!(p,0)
    jj = Threads.Atomic{Int}(0)
    l = Threads.SpinLock()

    Threads.@threads for k in 1:ntree
        nodes_temp, Vsplit_temp, Centres_temp, pred_temp, boot_temp = ERtmax(Im,Y,mtry,ntry, minElem,dist)
        nodes[1:length(nodes_temp),k] = nodes_temp
        Vsplit[1:length(Vsplit_temp),k] = Vsplit_temp
        centres[1:size(Centres_temp,1),:,k] = Centres_temp
        P[:,1:size(pred_temp,2),k]= pred_temp
        boot[k,1:length(boot_temp)]= boot_temp
        ## on affiche la barre de progression ::
        Threads.atomic_add!(jj, 1)
        Threads.lock(l)
        update!(p, jj[])
        Threads.unlock(l)
    end

    return nodes, Vsplit, centres, P, boot
end


