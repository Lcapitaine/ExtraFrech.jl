# ExtraFrech

[![Build Status](https://travis-ci.com/Lcapitaine/ExtraFrech.jl.svg?branch=master)](https://travis-ci.com/Lcapitaine/ExtraFrech.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/Lcapitaine/ExtraFrech.jl?svg=true)](https://ci.appveyor.com/project/Lcapitaine/ExtraFrech-jl)
[![Coverage](https://codecov.io/gh/Lcapitaine/ExtraFrech.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/Lcapitaine/ExtraFrech.jl)
[![Coverage](https://coveralls.io/repos/github/Lcapitaine/ExtraFrech.jl/badge.svg?branch=master)](https://coveralls.io/github/Lcapitaine/ExtraFrech.jl?branch=master)


This is an example of Fréchet extremely random forest :
`using ExtraFrech, Distances, Statistics`
`X=rand(20,100,10)` Simulate a database with 10 covariates of 100 individuals (curves or images) of length 20.
`Y= 2.* log.(1 .+ mean(X[:,:,1], dims=2))` Response variable.
`dist = Euclidean()` Set up a distance, see `Distances.jl` for all usable distances.
`@time frf, P, boot = ERFRF(X, Y, 3, 500, 5, dist)` Build an Extremely random forest with 500 trees, `mtr=3` and `ntry=5`.
