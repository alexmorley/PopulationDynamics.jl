module PopulationDynamics
using Reexport
@reexport using EPhys
using MultivariateStats, StatsBase

export fit!, weights

include("types.jl")
include("utils.jl")
include("pcaica.jl")

#include("coactivation.jl")
#include("coactivation_surrogates.jl")
#include("crosscor.jl")

end 
