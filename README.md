# ExtraFrech

[![Build Status](https://travis-ci.com/Lcapitaine/ExtraFrech.jl.svg?branch=master)](https://travis-ci.com/Lcapitaine/ExtraFrech.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/Lcapitaine/ExtraFrech.jl?svg=true)](https://ci.appveyor.com/project/Lcapitaine/ExtraFrech-jl)
[![Coverage](https://codecov.io/gh/Lcapitaine/ExtraFrech.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/Lcapitaine/ExtraFrech.jl)
[![Coverage](https://coveralls.io/repos/github/Lcapitaine/ExtraFrech.jl/badge.svg?branch=master)](https://coveralls.io/github/Lcapitaine/ExtraFrech.jl?branch=master)


This is an example of Fréchet extremely random forest :

# Arguments
* `X`: the matrix of the explanatory variables.
* `Y`: the output scalar.

Load the packages:

```
julia
julia> using ExtraFrech, Distances, Statistic
```

We simulate inputs `X` with 10 covariates of 100 individuals (curves or images) of length 20 and the associated output `Y`.

```
julia> X=rand(20,100,10)
julia> Y= 2.* log.(1 .+ mean(X[:,:,1], dims=2))
```

# Fréchet extremely random forest

Form : `ERFRF(X,Y,mtry,ntree,ntry,dist)`


* `X`: the matrix of the explanatory variables.
* `Y`: the output.
* `mtry`: number of variables randmly tested at each split.
* `ntree`: number of Fréchet extremely randomized trees built.
* `ntry`: number of centers randomly selected at each split.
* `dist`: distance used (see `Distances.jl`)

```
julia> dist = Euclidean()
julia> @time frf, P, boot = ERFRF(X, Y, 3, 500, 5, dist)
```
