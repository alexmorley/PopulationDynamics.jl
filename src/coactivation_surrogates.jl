
function detectICAensembles_surrogate(binnedspikematrix, binsize, npatterns)
    nbins = size(binnedspikematrix,1)
    surrogate_bins = sample(1:nbins,nbins,replace=true)
    surrogate_spikematrix = view(binnedspikematrix,surrogate_bins,:)
    _,_,_,_,surV = detectICAensembles(binnedspikematrix, binsize, npatterns)
    return surV
end

function trackdetectensemble_surrogates(metadata, sessions, use_cache=true;
    only_cache=false, nsurrogates = 5000)
    if use_cache && EPhys.checkcache(metadata,sessions)
        ensemble, surrogates = loadcache_surrogates(metadata,sessions)
        return ensemble, surrogates
    elseif only_cache
        error("Cached results not found")
    end
    # Bin Matrix
    cells2use = [false; metadata["cellIDs"].!="lick"]
    binsize = 0.02 #secs
    spiketimes = loadspikes(SpikeTimes, metadata, sessions)
    spikematrix = loadspikes(SpikeMatrix, metadata, sessions)
    spiketimes = getindex.(spiketimes,find(cells2use))
    binsamples = binsize*spiketimes[1].fs
    binnedspiketimes = fit.([Histogram], times.(spiketimes),
        [0:binsamples:spiketimes[1].maxtime], closed=:right)
    binnedspikematrix = cat(2,getfield.(binnedspiketimes,[:weights])...);

    Z, outPCA, npatterns, outICA, V = 
        EPhys.detectICAensembles(binnedspikematrix, binsize)
    surrogateVs = pmap(x->detectICAensembles_surrogate(binnedspikematrix,
            binsize, npatterns), 1:nsurrogates);
    reorder!.(surrogateVs,[V])
    Zconv, R, activationtimes = EPhys.trackensemble(spikematrix, V, binsize, cells2use)
    ensemble = ICAEnsemble(Z, npatterns, outPCA, outICA, V, Zconv, R, activationtimes)

    try
        cacheensemble_surrogates(metadata, sessions, ensemble, surrogateVs)
    catch
        warn("Cache Failed.")
    end
    return ensemble,surrogateVs
end


using JLD, Munkres
function reorder!(tst,V)
    cormat = cor(V, tst)
    sur_perm = munkres(-cormat)
    tst .= tst[:,sur_perm]
end

function cacheensemble_surrogates(metadata::Dict, sessions, 
        ensemble::ICAEnsemble, surrogates::Array{Array{Float64,2},1})
    filename = "$(metadata["bsnm"])_$(sessions...).icaensemble"
    saveloc = "$(EPhys.tempfiledir)/$(metadata["bsnm"])/$filename.jld"
    mkpath("$(EPhys.tempfiledir)/$(metadata["bsnm"])")
    jldopen(saveloc, "w") do file
        write(file, "ensemble", ensemble)
        write(file, "surrogates", surrogates)
    end
end

function loadcache_surrogates(metadata,sessions)
    filename = "$(metadata["bsnm"])_$(sessions...).icaensemble"
    cacheloc = "$(EPhys.tempfiledir)/$(metadata["bsnm"])/$filename.jld"
    cache = JLD.load(cacheloc)
    return cache["ensemble"], cache["surrogates"]
end
