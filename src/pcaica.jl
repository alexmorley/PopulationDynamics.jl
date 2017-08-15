export PCAICA

mutable struct PCAICA <: PopulationModel
	k::Int
	pca::MultivariateStats.PCA
	ica::MultivariateStats.ICA
end

PCA() = MultivariateStats.PCA{T=Float64}(Array{T,1}(),Array{T,2}(0,0),Array{T,1}(),
                                         T(NaN),T(NaN))
ICA() = MultivariateStats.ICA(Array{Float64,1}(),Array{Float64,2}(0,0))
PCAICA() = PCAICA(0, PCA(), ICA())
PCAICA(k::Int) = PCAICA(k, PCA(), ICA())

function fit!{T<:Float64}(model::PCAICA, Z::Array{T,2})
	model.pca = fit(MultivariateStats.PCA, Z)
	λmin,λmax = marchenko_thresh(size(Z)...)
	model.k = model.k == 0 ? sum(model.pca.prinvars.>λmax) : model.k
	Psign = projection(model.pca)[:,1:model.k] # projection of first k pc's
	Zproj = At_mul_B(Psign, Z) # project spikes onto pc subspace
	model.ica = fit(MultivariateStats.ICA, Zproj, model.k, maxiter=1000)
end

function weights(model::PCAICA)
	Psign = projection(model.pca)[:,1:model.k]
	V = Psign * model.ica.W
	normweightvectors!(V)
	return V
end	
