using Images

export fit_spike_shift

## these functions rely on the Ephys IO format for metadata & loading in spike times.

function fit!(model::PopulationModel, metadata::Dict, sessions,
		rates=(0.05,10.))
    spiketimes = loadspikes(SpikeTimes, metadata, sessions)
    cells2use = EPhys.ratefilt(metadata["cellIDs"], spiketimes, rates...)[2:end]
    Z = binZ(spiketimes)[:,cells2use]'
    fit!(model, Z)
    return model, Z
end

function fit!(model_type::Type, all_metadata::Array{<:Dict,1}, all_sessions;
		N=1000)
    Zs = []
    models = []
    btstrps = []

    for (metadata, sessions) in zip(all_metadata, all_sessions)
        model,Z = fit!(model_type(), metadata, sessions)
        btstrp = bootstrap(model, Z, N)
        push!(Zs     , Z)
        push!(models , model)
        push!(btstrps, btstrp)
    end
    return Zs,models,btstrps
end

function track_model(model_t, metadata::Dict, spiketimes::SpikeTimes;
                    ratefilt::Tuple{Float64,Float64} = (0.05,10.),                   
                    binsize::Float64 = 0.02,
                    tracking_resolution::Float64 = 0.001,
                    tracking_kernelwidth::Float64 = sqrt(binsize/
                                                         tracking_resolution),
                    tracking_type::Symbol = :full,
                    apply_tetrode_mask = true
                )
    cells2use = EPhys.ratefilt(metadata["cellIDs"], spiketimes,ratefilt[1],
                               ratefilt[2])[2:end]
    if apply_tetrode_mask
        # we don't want to track cells from the same tetrode together as 
        # otherwise we might be biased by spike sorting errors
        mask   = [x==y for x in metadata["tetlist"][cells2use],
                     y in metadata["tetlist"][cells2use]]
    else
        mask   = [x==y for x in 1:sum(cells2use), y in 1:sum(cells2use)]
    end

    Z = binZ(spiketimes, binsize)[:,cells2use]'
    Zconv = binZ(spiketimes, tracking_resolution)[:,cells2use]
    Zconv = imfilter(Zconv, KernelFactors.IIRGaussian([tracking_kernelwidth;0]))
    
    model = model_t()
    PopulationDynamics.fit!(model, Z)
                
    if tracking_type == :full
        R_split = [track(Zconv', weights(model), mask)]
    elseif tracking_type == :regional
        idmatch = cat([contains.(metadata["cellIDs"][cells2use]) for id in regions]...,dims=2)
        multipliers = [BitArray(idmatch[:,i] * idmatch[:,j]') 
            for i in indices(idmatch,2) for j in indices(idmatch,2)]
        R_split = track_partial(Zconv, weights(model), multipliers)
    elseif tracking_type == :diagvsoff
        idmatch = cat([occursin.(id,metadata["cellIDs"][cells2use]) for id in regions]..., dims=2)
        multiplier_ = BitArray(idmatch * idmatch')
        multipliers = [multiplier_, .!(multiplier_)]
        R_split = track_partial(Zconv, PopulationDynamics.weights(model), multipliers)
    else
        error("""
        No tracking type $tracking_type:
        Should be one of:
            - full
            - regional
            - diagvsoff
        """)
    end
    #@warn "This now returns both the model, the tracked trace, and the binned spike times"
    return model, R_split, Z
end

function fit_spike_shift(model_type::Type, metadata, sessions, jitters, nreps=10;
    verbose=false)
    # detect "real ensembles"
    spiketimes = loadspikes(SpikeTimes, metadata, sessions)
    cells2use = EPhys.ratefilt(metadata["cellIDs"], spiketimes, 0.05, 10.)[2:end]
    Z = binZ(spiketimes)[:,cells2use]'
    model = model_type()
    fit!(model, Z)
    
    # for each mean jitter jit
    models_alljitters = []
    for jitter in jitters
        verbose && println("Refitting with jitters in the range of $(jitter[1])-$(-jitter[1])")
        verbose && tic()
        models_jitter = pmap(_ -> shift_refit(model, spiketimes, jitter, cells2use), 1:nreps)
        push!(models_alljitters, models_jitter)
        verbose && toc()
    end
    model, models_alljitters
end

function shift_refit(model, spiketimes, jitter, cells2use)
    model_shifted = typeof(model)(model.k, pca(), ica(), srand())
    # shift spikes
    spiketimes_shifted = EPhys.addjitter(spiketimes, jitter)
    Z_shifted = binZ(spiketimes_shifted)[:,cells2use]' # this is hacky, "spiketimes should already be filtered"
    # re-detect
    fit!(model_shifted, Z_shifted)
    return model_shifted
end 
