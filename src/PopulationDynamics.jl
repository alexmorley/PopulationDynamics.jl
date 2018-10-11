module PopulationDynamics
using Reexport
using utils
@reexport using EPhys

if VERSION >= v"0.7.0"
    using Random
    using Statistics
    using LinearAlgebra
end

using MultivariateStats, StatsBase
using RecursiveArrayTools, TensorDecompositions, Munkres


export fit!, weights
export track, track_partial, bootstrap, confint
export similarity, stability
export factors

include("types.jl")
include("utils.jl")
include("pcaica.jl")
include("tensor_decompositions.jl")

include("ephysIO.jl")
#include("coactivation.jl")
#include("coactivation_surrogates.jl")
#include("crosscor.jl")

end 
