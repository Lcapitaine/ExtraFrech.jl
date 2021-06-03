module ExtraFrech

# Write your package code here.
using Distances
using Statistics
using StatsBase
using Random
using ProgressMeter



include("ERFRF.jl")
include("ERtmax.jl")
include("ERvar_split.jl")
include("FRFERR.jl")
include("Importance.jl")
include("Impurity_split.jl")
include("Impurity.jl")
include("Pred_rf.jl")
include("Pred_tree.jl")


end
