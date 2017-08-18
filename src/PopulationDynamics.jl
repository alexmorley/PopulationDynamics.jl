module PopulationDynamics
using Reexport
@reexport using EPhys
using MultivariateStats, StatsBase
using TensorDecompositions

export fit!, weights, track, bootstrap, confint

include("types.jl")
include("utils.jl")
include("pcaica.jl")
include("tensor_decompositions.jl")

#include("coactivation.jl")
#include("coactivation_surrogates.jl")
#include("crosscor.jl")

end 
