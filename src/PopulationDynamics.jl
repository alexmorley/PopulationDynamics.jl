module PopulationDynamics
using Reexport
@reexport using EPhys
using MultivariateStats, StatsBase

include("types.jl")
include("coactivation.jl")
include("coactivation_surrogates.jl")
#include("crosscor.jl")

end 
