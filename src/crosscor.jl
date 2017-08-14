getcompweights(comp::Array{Float64,1}, crosscors::Array{Float64,3}) = 
    reshape(cor(comp, crosscors[:,:]), size(crosscors,2,3))

function getweights(outICA, allcrosscors, nan2zeros=false, normalise=true)
    num_components = size(outICA.W,2) 
    compweights = [getcompweights(outICA.W[:,x],allcrosscors) for x in 1:num_components]
    nan2zeros && nan2zero!.(compweights)
    normalise && zscore!.(compweights, mean.(compweights),std.(compweights))
    return compweights
end

function getfit(stat, crosscors, num_components)
    @show myid(), num_components
    statfit = fit(stat, convert(Array{Float64,2}, crosscors), num_components, verbose=false, maxiter=10000);
end

function getfitsummary(stat, crosscors, num_components)
    fit = getfit(stat, crosscors, num_components)
    getfitsummary(fit, crosscors)
end

function getfitsummary(fit::MultivariateStats.ICA, crosscors)
    sum(abs(getweights(fit, crosscors)), (2,3))
end

function getbestexamples(weights, n, num_components)
    map(com->selectperm(weights[com,:,:][:], 1:n, rev=false),1:num_components)
end

significantcellpairs(args...) = significantweights(args...)[2]

function significantweights(outICA, allcrosscors, threshold)
    weights = EPhys.getweights(outICA, allcrosscors, true)
    [(diagind(x) = 0) for x in weights]
    cellpairs = (x->abs.(x).>threshold).(weights)
    
    threshweights = deepcopy(weights)
    [x[!c]=0. for (x,c) in zip(threshweights,cellpairs)]
    return threshweights, cellpairs
end

function getstrength!(out,pattern,crosscors)
    @inbounds for ind in 1:size(crosscors,2)
        out[ind] += dot(view(crosscors,:,ind),pattern)
    end
end

function weightcors!{T<:Real}(cors, weight::T)
    (weight == 0) && (fill!(cors,0);return)
    @inbounds for i in eachindex(cors)
        cors[i] *= weight
    end
end

getstrengthpoint(args) = getstrengthpoint(args...)

function getstrengthpoint(ICw, spikes::SpikeTimes, lags, cellpairs2use::BitArray)
    #pre-allocate
    activation = spzeros(spikes.spiketimes[1].maxtime, sum(cellpairs2use))
    pointscor = zeros(Float64, length(lags), 
                    maximum(map(x->length(x),spikes.spiketimes)))
    cpind=0
	# for each cell pair
    #@showprogress for (ind,spiketimes) in enumerate(spikes.spiketimes)
    #    for (ind2,spiketimes2) in enumerate(spikes.spiketimes)
    @showprogress for (ind,ind2) in zip(findn(cellpairs2use)...)
		cpind += 1       
        # get view of temp array
        pc = view(pointscor,:, eachindex(spikes.spiketimes[ind].timestamps))
        # get instantaneous cross-correletion
        slpxcorr!(pc,spikes.spiketimes[ind2].timestamps,
			spikes.spiketimes[ind].timestamps,lags)
        # dot product of component pattern with instant xcorr 
        getstrength!(
			view(activation,floor(Int,spikes.spiketimes[ind].timestamps),cpind),
			ICw, pc)
        # reset pre-allocated xcorr array
        view(pointscor, :, 1:length(spikes.spiketimes[ind].timestamps)) .= 0
    end
    activation::SparseMatrixCSC{Float64,Int64}
end


"""
    getassemblyactivationpoints(assemblyactivations, thresh)
Get discrete times of activation from continuous activation strength.    
"""
function getassemblyactivationpoints(assemblyactivations, thresh)
    ispeak = map(falses, assemblyactivations)
    [(y[getindex.(findlocalmaxima(x),1)]=true) for (x,y) in zip(assemblyactivations,ispeak)]
    
    isoverthresh =  map(falses,assemblyactivations)
    [(y[zscore(x).>thresh]=true) for (x,y) in zip(assemblyactivations,isoverthresh)]
    
    isactivationpeak = [x & y for (x,y) in zip(ispeak, isoverthresh)]
    return isactivationpeak
end

"""
	function parallel_fiteachcomp(crosscorsIN, min_comp, max_comp, stat=ICA)	
Applies ICA with N components where N is a unit range
"""
function parallel_fiteachcomp(processors, crosscorsIN, min_comp, max_comp, stat=ICA) 
    fits = pmap(processors, x->getfit(stat, crosscorsIN,x), min_comp:max_comp)
    return fits
end

"""
	fitNICAComponentsCrosscors([processors,] spiketimes, bins; normalise_crosscors=true,
	    usediag=false, just_lower=true, min_comp=3, max_comp=20, verbose=false)	
- Cacluates Cross Correlegrams
- Selects which ones to use
- Applies ICA with N components where N is a unit range
"""
function calc_fit_crosscors(spiketimes::SpikeTimes; kwargs...)
    allcrosscors, corselected = calc_select_crosscors(spiketimes::SpikeTimes; kwargs...)
    outICA, outPCA, num_components = pcaica(allcrosscors[:,corselected])
    return allcrosscors, corselected, outICA, outPCA, num_components
end

function calc_select_crosscors(spiketimes::SpikeTimes;
    bins=-250:5:250, normalise_crosscors=true, usediag=false, just_lower=true,
    verbose=false, cells2use=:, intrapairstoignore=[], _ignore...)

    
    verbose && println("Getting Crosscorrelations...")
    allcrosscors = getallcrosscors(spiketimes, bins, normalise_crosscors,
        cells2use)
    
    verbose && println("Selecting Crosscorrelations...")
    excludecors = falses(allcrosscors[1,:,:])
    excludecors[intrapairstoignore,intrapairstoignore] = true
    corselected = selectcrosscors!(allcrosscors,
                                  usediag, just_lower,
                                  excludecors[:])
    
    return allcrosscors, corselected
end

function pcaica(matrix)
    outPCA,λmax,num_components = EPhys.pca_npatterns(matrix)
    outICA = EPhys.getfit(ICA, matrix, num_components)
    return outICA, outPCA, num_components
end


if false
    function fitNICAComponents2ShufCrosscors(processors::Base.AbstractWorkerPool, orig_spikematrix,
        nshuf::Int; kwargs...)     
        shufICAComponentfits = Array{Tuple{Array{Float64,3},Array{Bool,1},Array{Any,1}},1}(nshuf)
        for i in 1:nshuf
            println("Shuffle $i of $nshuf")
            shufspikematrix = mapslices(shuffle, orig_spikematrix, 1)
            shufspiketimes = getspiketimesarray(shufspikematrix)
            shufICAComponentfits[i] = fitNICAComponents2Crosscors(default_worker_pool(),
                shufspiketimes; kwargs...)
        end
        return shufICAComponentfits
    end 
end

"""
	function getstrength(pattern,crosscors[,weights])
Get the strength of a particular pattern for a set of crosscorrelations
Use `weights` to weight output by a matrix with the same dims as crosscors
"""
function getstrength(pattern,crosscors) 
    tweights = mapslices(x->dot(x,pattern),crosscors,1)
end

function getstrength(pattern,crosscors, weights) 
    tweights = mapslices(x->dot(x,pattern),crosscors,1).*weights
    sum(tweights)::Float64
end
"""
	function ptrack_ensemble{T}(processors, spikematrix, windows, ,bins, ICw::Array{T,2},
	Cweights::Array{T,3}; use_proj=true, batch_size=1, temp="none")
Track assembly strength over time using multiple processors.
NB This function is still under development
"""
function ptrack_ensemble{T}(processors, spikematrix, ICw::Array{T,2}, ICweights::Array{T,3};
    use_proj=true, bins=-200:5:200, batch_size=10, windows=0,
    verbose=false, metadata=Dict(), ignore_...)

    options = Dict(:use_proj=>use_proj, :batch_size=>batch_size, :metadata=>metadata, :bins=>bins,
    :windows=>windows)
    try
        num_components = size(ICw,2)
        trackedensembles = zeros(length(windows), num_components)
        verbose && println("Windowing Spikes")
        
        sliding_spikes = pmap(processors, win->getspikes(spikematrix, win), windows,
        batch_size=100);
        
        verbose && println("Computing Sliding Crosscorrelations") 
        #chunksize = length(processors.workers)*batch_size
        chunksize = 100
        @show length(processors.workers),batch_size,chunksize
        (length(windows) < chunksize) && (chunksize = length(windows))
        time_created = string(Dates.format(now(),"yymmdd_HHMMSS")) 
        verbose && println("Using groupname var_$time_created")

        sliding_cor_chunk = [zeros(Int, length(bins), size(spikematrix,2), size(spikematrix,2))
        for x in 1:chunksize]
        scor = zeros(Int, length(bins), size(spikematrix,2), size(spikematrix,2), chunksize)
        
        @show length(sliding_spikes)
        @showprogress for index in 1:chunksize:length(sliding_spikes)
            ## Chunking
            if index < (length(sliding_spikes)-chunksize)
                chunkrange = index:(index+chunksize-1)
            else
                chunkrange = index:length(sliding_spikes)
            end
            println("Processing chunk $chunkrange of $(length(sliding_spikes))")
            ## Sliding Cross Cor
            sliding_cor_chunk[1:length(chunkrange)] = pmap(processors, spks->allcross_mat(spks, bins), sliding_spikes[chunkrange],
            batch_size=batch_size)
            for (ind, val) in enumerate(sliding_cor_chunk[1:length(chunkrange)])
                scor[:,:,:,ind] = val
            end

            ## Track Portion
            println("Projecting Chunk Onto Components...") 
            @showprogress for (ind,chunkbit) in enumerate(chunkrange)
                for component in 1:num_components
                    trackedensembles[chunkbit, component] = getstrength(ICw[:,component],scor[:,:,:,ind],
                    ICweights[component,:,:])
                end
            end
        end
        
        println("Saving Tracked Ensembles...")
        savebigfile(metadata["bsnm"], "ensemble_tracked", [trackedensembles],
            ["trackedensembles"], options)
        return trackedensembles
    catch error
        @everywhere gc(true)
        sleep(0.5)
        @everywhere gc(true)
        throw(error)
    end
end


#function projensemblecor{T}(processors, bsnm, loadoptions; use_proj=true)
#loadbigfile(bsnm, "assembly", ["outICA", "weights"], loadoptions)
#loadbigfile(bsnm, "slidingcor", "sliding_cor", loadoptions, memorymap=true)
#i 
#end 


function projensemblecor{T}(processors, sliding_cor, ICw::Array{T,2},
    ICweights::Array{T,3}; use_proj=true, batch_size=10, metadata=0, bins=0, windows=0, _ignore...)
    
    options = Dict(:use_proj=>use_proj, :batch_size=>batch_size, :metadata=>metadata, :bins=>bins,
    :windows=>windows)
    
    num_components = size(ICw,2)
    @show processors
    println("Projecting Component Weights back onto Sliding CrossCorrelatons")         
    trackedensembles = projensemblecor_main(sliding_cor, 
        ICw, ICweights, num_components, batch_size, use_proj)

     savebigfile(metadata["bsnm"], "ensemble_tracked", [trackedensembles],
        ["trackedensembles"], options)
    return trackedensembles    
end

function projensemblecor_main(sliding_cor, ICw, ICweights, num_components,
    batch_size,use_proj)
    if use_proj
        out = zeros(num_components, size(sliding_cor,4))
        for index in 1:batch_size:size(sliding_cor,4)
            if index < (size(sliding_cor,4)-batch_size)
                chunkrange = index:(index+batch_size-1)
            else
                chunkrange = index:size(sliding_cor,4)
            end
            println("Processing $chunkrange...")
            @showprogress for chunkbit in chunkrange
                scor = sliding_cor[:,:,:,chunkbit]
                for component in 1:num_components
                    out[component, chunkbit] = getstrength(ICw[:,component],scor,
                    ICweights[component,:,:])
                end
            end
            
            #out[chunkrange] = pmap(processors, y->getstrength(ICw,y,ICweights),
            #sliding_cor[:,:,:,chunkrange])
        end
        return out
    else
        return pmap(processors, y->getstrength(ICw,y), sliding_cor)
    end
end


#=
function ptrack_ensemble{T}(spikematrix, windows, bins, ICw::Array{T,2},
    ICweights::Array{T,3}; use_proj=true)
    num_components = size(ICw,2)
    trackedensembles = zeros(length(windows), num_components)

    for component in 1:num_components
        @show component
        trackedensembles[:,component] = projensemblecor(spikematrix, windows,
            bins, ICw[:,component], ICweights[component,:])
        sleep(0.5)
    end
    trackedensembles
end=#

"""
	function ptrack_ensemble{T}(processors, spikematrix, windows, bins, ICw::Array{T,2},
	Cweights::Array{T,3}; use_proj=true, batch_size=1, temp="none")
Track assembly strength over time using multiple processors.
NB This function is still under development
"""
function ptrack_ensemble_old{T}(processors, spikematrix, ICw::Array{T,2}, ICweights::Array{T,3};
    use_proj=true, bins=-200:5:200, batch_size=10, windows=0,
    verbose=false, metadata=Dict(), ignore_...)

    options = Dict(:use_proj=>use_proj, :batch_size=>batch_size, :metadata=>metadata, :bins=>bins,
    :windows=>windows)
    try
        num_components = size(ICw,2)
        trackedensembles = zeros(length(windows), num_components)
        verbose && println("Windowing Spikes")
        
        sliding_spikes = pmap(processors, win->getspikes(spikematrix, win), windows,
        batch_size=100);
        
        verbose && println("Computing Sliding Crosscorrelations") 
        #chunksize = length(processors.workers)*batch_size
        chunksize = 100
        @show length(processors.workers),batch_size,chunksize
        (length(windows) < chunksize) && (chunksize = length(windows))
        time_created = string(Dates.format(now(),"yymmdd_HHMMSS")) 
        verbose && println("Using groupname var_$time_created")

        sliding_cor_chunk = [zeros(Int, length(bins), size(spikematrix,2), size(spikematrix,2))
        for x in 1:chunksize]
        sliding_cor_chunk_mat = zeros(Int, length(bins), size(spikematrix,2), size(spikematrix,2), chunksize)
        
        @show length(sliding_spikes)
        @showprogress for index in 1:chunksize:length(sliding_spikes)
            if index < (length(sliding_spikes)-chunksize)
                chunkrange = index:(index+chunksize-1)
            else
                chunkrange = index:length(sliding_spikes)
            end
            println("Processing chunk $chunkrange of $(length(sliding_spikes))")
            sliding_cor_chunk[1:length(chunkrange)] = pmap(processors, spks->allcross_mat(spks, bins), sliding_spikes[chunkrange],
            batch_size=batch_size)
            println("Writing chunk ...")
            for (ind, val) in enumerate(sliding_cor_chunk[1:length(chunkrange)])
                sliding_cor_chunk_mat[:,:,:,ind] = val
            end
            savebigfile(metadata["bsnm"], "slidingcor", [sliding_cor_chunk_mat[:,:,:,1:length(chunkrange)]], ["sliding_cor"],
            time_created, (:,:,:,chunkrange,), (size(sliding_cor_chunk_mat)[1:3]...,length(sliding_spikes)),
            options)
        end

        println("sliding cors saved")
        return nothing
    catch error
        @everywhere gc(true)
        sleep(0.5)
        @everywhere gc(true)
        throw(error)
    end
end
