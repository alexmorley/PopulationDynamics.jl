module PopulationDynamics
using Reexport
using utils
@reexport using EPhysBase

if VERSION >= v"0.7.0"
    using Random
    using Statistics
    using LinearAlgebra
end

using MultivariateStats, StatsBase
using RecursiveArrayTools, Munkres
#TensorDecompositions

export fit!, weights
export track, track_partial, bootstrap, confint
export similarity, stability
export factors

abstract type PopulationModel end

include("utils.jl")
include("pcaica.jl")
include("types.jl")
#include("tensor_decompositions.jl")

include("ephysIO.jl")
#include("coactivation.jl")
#include("coactivation_surrogates.jl")
#include("crosscor.jl")

end 
