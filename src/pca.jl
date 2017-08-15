export PCA

mutable struct PCA <: PopulationModel
    k::Int
    pca::MultivariateStats.PCA
end


