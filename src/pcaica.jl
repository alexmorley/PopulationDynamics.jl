export PCAICA, PCA #, FA

ica() = MultivariateStats.ICA(Array{Float64,1}(),
                              Array{Float64,2}(undef,0,0))
pca() = MultivariateStats.PCA(Array{Float64,1}(),
                              Array{Float64,2}(undef,0,0),
                              Array{Float64,1}(),
                              Float64(NaN),
                              Float64(NaN))
"""
mutable struct PCA <: PopulationModel
Principal Components Analysis
"""
mutable struct PCA <: PopulationModel
    k::Int
    pca::MultivariateStats.PCA
    function PCA(k=0::Int, pca=pca())
        new(k,pca)
    end
end

function fit!(model::PCA, Z::AbstractArray; use_marchenko=false)
    model.pca = fit(MultivariateStats.PCA, Z)
    if use_marchenko
        λmin,λmax = marchenko_thresh(size(Z)...)
        model.k = model.k == 0 ? sum(model.pca.prinvars.>λmax) : model.k
    else
        model.k = length(model.pca.prinvars)
    end
    return model
end

function weights(model::PCA)
    V = projection(model.pca)[:,1:model.k]
    normweightvectors!(V)
    return V
end

"""
mutable struct PCAICA <: PopulationModel
See: Detecting cell assemblies in large neuronal populations. Lopes dos Santos et al 2013.
"""
mutable struct PCAICA <: PopulationModel
    k::Int
    pca::MultivariateStats.PCA
    ica::MultivariateStats.ICA
    seed::AbstractRNG
    function PCAICA(k=0::Int, pca=pca(), ica=ica(), seed=Random.seed!(13))
        new(k, pca, ica, seed)
    end
end

function fit!(model::PCAICA, Z::AbstractArray) 
    model.pca = fit(MultivariateStats.PCA, Z)
    λmin,λmax = marchenko_thresh(size(Z)...)
    model.k = model.k == 0 ? sum(model.pca.prinvars.>λmax) : model.k
    Psign = projection(model.pca)[:,1:model.k] # projection of first k pc's
    Zproj = transpose(Psign) * Z # project spikes onto pc subspace
    model.ica = fit(MultivariateStats.ICA, Zproj, model.k,
                    winit=rand(model.seed, size(Zproj,1), model.k), # for reproducibility
                    maxiter=10000, tol=1.0e-6)
end

function weights(model::PCAICA)
    Psign = projection(model.pca)[:,1:model.k]
    V = Psign * model.ica.W
    normweightvectors!(V)
    return V
end
