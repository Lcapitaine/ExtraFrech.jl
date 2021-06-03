function ERvar_split(Y::Vector{Float64},Im::Array{Float64,3}, ntry::Int64, dim::Tuple{Int,Int,Int}, dist)

    l::Int64=dim[2]

    if l==2

        D::Vector{Float64} = zeros(dim[3])
        for p in 1:dim[3]
            D[p]= mean(sqrt.((Im[:,1,p]-Im[:,2,p]).^2))
        end
        return findmax(D)[2], [1, 2], [1 ,2], 0

    else
        impur::Vector{Float64} = zeros(dim[3])
        Splits_prime = Array{Int}(undef,dim[3],dim[2])
        centers = Array{Int}(undef,2, dim[3])
        whichsplit::Int64 = 0
        n_dec::Int64 = min(ntry,l)
        centers_prime = Array{Int64}(undef,2,n_dec)


        for p in 1:dim[3]

            ## plus de tests que d'images ??
            splits::Array{Float64,2} = ones(n_dec, l)
            impur_prime = Vector{Float64}(undef, n_dec)

            for k in 1:n_dec
                centers_prime[:,k] = sample(1:dim[2], 2, replace=false)
                d_g = Distances.colwise(dist, Im[:,:,p], Im[:,centers_prime[1,k],p]) ## distance à gauche
                d_d = Distances.colwise(dist, Im[:,:,p], Im[:,centers_prime[2,k],p]) ## le fameux dd
                qui = findall(x->x>=0, d_g-d_d)::Vector{Int}
                splits[k,qui].=  2
                impur_prime[k]= impurity_split(Y,splits[k,:])::Float64
            end

            impur[p], whichsplit = findmin(impur_prime)
            Splits_prime[p,:] = @view splits[whichsplit,:]
            centers[:,p]= @view centers_prime[:,whichsplit]
        end

        Imp, whichsplit = findmin(impur)
        return whichsplit, centers[:,whichsplit], Splits_prime[whichsplit,:], Imp
    end

end
