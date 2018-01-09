export fit_spike_shift

## these functions rely on the Ephys IO format for metadata & loading in spike times.

function track_model(model_t, metadata::Dict, spiketimes::SpikeTimes;
                    ratefilt::Tuple{Float64,Float64} = (0.05,10.),                   
                    binsize::Float64 = 0.02,
                    tracking_resolution::Float64 = 0.001,
                    tracking_kernelwidth::Float64 = sqrt(20),
                    tracking_type::Symbol = :full
                )
                
    cells2use = EPhys.ratefilt(metadata["cellIDs"], spiketimes,ratefilt[1], ratefilt[2]
                    )[2:end]

    Z = binZ(spiketimes, binsize)[:,cells2use]'
    Zconv = binZ(spiketimes, tracking_resolution)[:,cells2use]
    Zconv = imfilter(Zconv, KernelFactors.IIRGaussian([tracking_kernelwidth;0]))
    
    model = model_t()
    PopulationDynamics.fit!(model, Z)
                
    if tracking_type == :full
        R_split = [track(Zconv', weights(model))]
    elseif tracking_type == :regional
        idmatch = cat(2,[contains.(metadata["cellIDs"][cells2use],id) for id in regions]...)
        masks = [BitArray(idmatch[:,i] * idmatch[:,j]') 
            for i in indices(idmatch,2) for j in indices(idmatch,2)]
        R_split = track_partial(Zconv, weights(model), masks)
    elseif tracking_type == :diagvsoff
        idmatch = cat(2,[contains.(metadata["cellIDs"][cells2use],id) for id in regions]...)
        mask_ = BitArray(idmatch * idmatch')
        masks = [mask_, .!(mask_)]
        R_split = track_partial(Zconv, PopulationDynamics.weights(model), masks)
    else
        error("""
        No tracking type $tracking_type:
        Should be one of:
            - full
            - regional
            - diagvsoff
        """)
    end
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
        models_jitter = pmap(_ -> shift_refit(model, spiketimes, jitter, cells2use), 1:nreps)
        push!(models_alljitters, models_jitter)
    end
    model, models_alljitters
end

function shift_refit(model, spiketimes, jitter, cells2use)
    model_shifted = typeof(model)(model.k)
    # shift spikes
    spiketimes_shifted = EPhys.addjitter(spiketimes, jitter)
    Z_shifted = binZ(spiketimes_shifted)[:,cells2use]' # this is hacky, "spiketimes should already be filtered"
    # re-detect
    fit!(model_shifted, Z_shifted)
    return model_shifted
end
