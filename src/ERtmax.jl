function ERtmax(Im::Array{Float64,3},Y::Vector{Float64},mtry::Int64,ntry::Int64, dim::Tuple{Int,Int,Int}, dist)
     ## On va éviter de calculer ça tout le temps !!!! A optimiser !!
    p::Int=dim[3]
    n::Int64= dim[2]
    boot::Vector{Int64} = unique(sample(1:n,n))
    V_split::Array{Float64,2} = zeros(4,2*length(boot)-1)
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

    for i in 1:length(boot)/2-1
        ## On va tirer un sous-échantillon de variables possibles ::
        feuilles_prime = feuilles
        if decoupe == length(unique(feuilles))
            break

        else
            for j in unique(feuilles)

                qui::Vector{Int}= collect(findall(x->x==j, feuilles))
                V::Vector{Int}= sort(sample(1:p, mtry, replace=false))

                ### Maintenant il faut se restreindre aux éléments dans la feuilles
                if length(qui)>=2
                    Split = ERvar_split(Y[boot][qui],X_boot[:,qui,V], ntry,(dim[1],length(qui),length(V)), dist)
                    V_split[1,courant] = j
                    V_split[2,courant] = V[Split[1]]
                    V_split[3,courant] = boot[qui[Split[2][1]]]
                    V_split[4,courant] = boot[qui[Split[2][2]]]

                    gauche::Vector{Int}= findall(x->x==1, Split[3])
                    droite::Vector{Int}= findall(x->x==2, Split[3])

                    feuilles_prime[qui[gauche]].= 2*j#*ones(length(gauche))
                    feuilles_prime[qui[droite]].= (2*j + 1)#*ones(length(droite))

                    Pred[1,courant_pred]=2*j
                    Pred[1,courant_pred+1]=2*j+1

                    Pred[2,courant_pred]= mean(Y[boot][qui[gauche]])
                    Pred[2,courant_pred+1] = mean(Y[boot][qui[droite]])

                    courant += 1
                    courant_pred +=2

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
    return V_split[:,1:courant], Pred[:,1:courant_pred-1], boot
end
