export fit_spike_shift

## these functions rely on the Ephys IO format for metadata & loading in spike times.

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
