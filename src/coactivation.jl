using MultivariateStats
using JLD

export ICAEnsemble, detectICAensembles, trackdetectensemble

#abstract type Assembly end

struct ICAEnsemble 
    Z::Array{Float64,2}
    K::Int
    PCAobj::PCA
    ICAobj::ICA
    V::Array{Float64,2}
    Zconv::Array{Float64,2}
    R::Array{Float64,2}
    activations::Array{Array{Int64,1},1}
end

marchenko_thresh(n,B) = ((1-sqrt(n/B))^2,(1+sqrt(n/B))^2)

"""
	function pca_npatterns(zbinnedmatrix)
Compute PCA on a Z-scored spike matrix (cells X bins) & return the PCA object, the
marchenko-pasteur threshold - λmax as well as the number of patterns that exceed
this threshold.
"""
function pca_npatterns(zbinnedmatrix)
    # compute PCA
    outPCA = fit(PCA, zbinnedmatrix)

    # get λ max
    n,B = size(zbinnedmatrix)
    λmin,λmax = marchenko_thresh(n,B)

    # N components with eigenvalues > λmax
    npatterns = sum(outPCA.prinvars.>λmax)
    return outPCA,λmax,npatterns
end

""" 
	pca_patterns_per_binsize(zbinnedmatrix, binsizes)
Useful to check how robust the number of patterns is to the choice of binsize.
"""
function pca_patterns_per_binsize(metadata,sessions,binsizes)
    npatterns = zeros(binsizes)
    spikematrix = nsessspikematrix(metadata, sessions) # 1ms bins
    for (ind,binsize) in enumerate(binsizes)
        binnedmatrix = binspikematrix(spikematrix, binsize)
        zbinnedmatrix = zscore(binnedmatrix,1)
        npatterns[ind] = pca_npatterns(zbinnedmatrix)[3]
    end
    return npatterns
end

"""
	function normweightvectors!(V)
Normalise the weight vector of each pattern so that the maximum absolute value
is one.
"""
function normweightvectors!(V)
    for i in 1:size(V,2)
        weightvec = view(V,:,i)
        if abs(minimum(weightvec)) > maximum(weightvec)
            weightvec .= -weightvec
        end
        weightvec ./= vecnorm(weightvec)
    end
end

"""
    function track(Z, V)
Track zscored firing rates Z using weight vectors V.
Tracking uses quadratic form
    R(t) = z(t)' * Pk(t) * (zt)
"""
function track(Z, V)
    npatterns = size(V,2)
    R = zeros(Float64, size(Z,1), npatterns)
    for k in 1:npatterns
        Rk = view(R,:,k)
        Pk = V[:,k]*V[:,k]'
        Pk[diagind(Pk)] .= 0.

        trackK!(Rk, Pk, Z)
    end
    return R
end

function trackK!(Rk,Pk,Z)
    for t in 1:size(Z,1)
        Rk[t] = Z[t,:]'*Pk*Z[t,:]
    end
    nothing
end



function detectICAensembles(binnedspikematrix, binsize, npatterns=-1)
    # Z-scored matrix (note Z is a bins * neuron matrix, as this is faster for computation)
    Z = zscore(binnedspikematrix,1)

    # Do PCA --> Extract number of patterns
    ### outPCA, λmax, npatterns = pca_npatterns(Z)
    ## compute PCA
    outPCA = fit(PCA, Z')
    ## get λ max
    n,B = size(Z')
    λmin,λmax = marchenko_thresh(n,B)
    ## N components with eigenvalues > λmax
    npatterns = npatterns == -1 ? sum(outPCA.prinvars.>λmax) : npatterns

    # Projection of first n principal components
    Psign = projection(outPCA)[:,1:npatterns]

    # Project Spike Matrix onto Subspace Spanned by First N PCA Components 
    Zproj = At_mul_Bt(Psign,Z)

    # ICA (unmixing matrix W)
    outICA = fit(ICA, Zproj, npatterns, maxiter=10000)

    # Express unmixing matrix W back into original subspace using Psign
    V = Psign * outICA.W;
    normweightvectors!(V)

    return Z, outPCA, npatterns, outICA, V
end
    
function trackensemble(spikematrix::SpikeMatrix, V::Array{Float64,2}, binsize, cells2use)
    # convolve spike matrix with gaussian kernel
    conv = utils.imfilter_gaussian(spikematrix.spikematrix[:,cells2use],
        [(spikematrix.fs*binsize)/sqrt(12); 0])#
    Zconv = zscore(conv,1)

    # track expression of assemblies in convolved spike matrix
    R = track(Zconv, V);#

    # find activation peaks
    # detect peaks in expression
    Rthresh = 4
    activationtimes = Array{Int,1}[]
    for i in 1:size(R,2)
        Rk = R[:,i]
        at = filter(x->Rk[x]>Rthresh,findlocalmaxima(Rk))
        push!(activationtimes, getindex.(at,[1]))
    end
    return Zconv, R, activationtimes
end

function trackdetectensemble(binnedspikematrix, binsize::Float64, spikematrix::SpikeMatrix, cells2use)
    Z, outPCA, npatterns, outICA, V = detectICAensembles(binnedspikematrix, binsize)
    Zconv, R, activationtimes = trackensemble(spikematrix, V, binsize, cells2use)
    return ICAEnsemble(Z, npatterns, outPCA, outICA, V, Zconv, R, activationtimes)
end

function trackdetectensemble(metadata::Dict,sessions,use_cache=true;
                            only_cache=false)
    cells2use = [false; metadata["cellIDs"].!="lick"]
    
    if use_cache && checkcache(metadata, sessions)
        return loadcache(metadata,sessions)
    elseif only_cache
        error("Cached results not found")
    end
    
    # Bin Matrix
    binsize = 0.02 #secs
    spiketimes = loadspikes(SpikeTimes, metadata, sessions)
    spikematrix = loadspikes(SpikeMatrix, metadata, sessions)
    spiketimes = getindex.(spiketimes,find(cells2use))
    binsamples = binsize*spiketimes[1].fs
    binnedspiketimes = fit.([Histogram], times.(spiketimes),
        [0:binsamples:spiketimes[1].maxtime], closed=:right)
    binnedspikematrix = cat(2,getfield.(binnedspiketimes,[:weights])...);
    ensemble = trackdetectensemble(binnedspikematrix, binsize, spikematrix,cells2use)
    try
        cacheensemble(metadata, sessions, ensemble)
    end
    return ensemble
end

function cacheensemble(metadata::Dict, sessions, ensemble::ICAEnsemble)
    filename = "$(metadata["bsnm"])_$(sessions...).icaensemble"
    saveloc = "$(tempfiledir)/$(metadata["bsnm"])/$filename.jld"
    mkpath("$(tempfiledir)/$(metadata["bsnm"])")
    jldopen(saveloc, "w") do file
        write(file, "ensemble", ensemble)
    end    
end

function checkcache(metadata,sessions)
    filename = "$(metadata["bsnm"])_$(sessions...).icaensemble"
    cacheloc = "$(tempfiledir)/$(metadata["bsnm"])/$filename.jld"
    isfile(cacheloc)
end

function loadcache(metadata,sessions)
    filename = "$(metadata["bsnm"])_$(sessions...).icaensemble"
    cacheloc = "$(tempfiledir)/$(metadata["bsnm"])/$filename.jld"
    JLD.load(cacheloc)["ensemble"]
end

function load(ens::Type{EPhys.ICAEnsemble}, metadata, sessions)
    trackdetectensemble(metadata,sessions,true,only_cache=true)
end

# conversions
function SpikeMatrix(ICAobj::ICAEnsemble)
    activations = ICAobj.activations
    patids = cat(1,ones.(Int,length.(activations)).*(1:length(activations))...)
    times = cat(1,activations...)
    return SpikeMatrix(EPhys.TimeStamps(times,0,maximum(times),1000.),
        TimeStamps(patids,0,maximum(times),1000.))
end

function SpikeTimes(ICAobj::ICAEnsemble)
    activations = ICAobj.activations
    patids = cat(1,ones.(Int,length.(activations)).*(1:length(activations))...)
    times = cat(1,activations...)
    return SpikeTimes(EPhys.TimeStamps(times,0,maximum(times),1000.),
                      TimeStamps(patids.+1,0,maximum(times),1000.), length(activations))
end
