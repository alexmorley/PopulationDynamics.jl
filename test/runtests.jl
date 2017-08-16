using PopulationDynamics
using Base.Test

### Load ### 

bsnm = "test"
datadir = abspath("test")
metadata = EPhys.getmetadata(bsnm, datadir)
sessions = [1];

spiketimes = loadspikes(SpikeTimes, metadata, sessions)
spiketimes = filter(x->length(x)>100, spiketimes)
Z = binZ(spiketimes)'

### Test ###

println("Testing Models...")
modelnames = ["PCAICA", "PCA"]
models = [PCAICA(), PCA()]
for (model,name) in zip(models, modelnames)
    println("   $name")
    println("       fit")
    fit!(model, Z)
    @test typeof(weights(model)) == Array{Float64,2}
    println("       track")
    @test track(model, zeros(size(Z,1),100)) == zeros(model.k,100)
    println("       bootstrap")
    @test typeof(bootstrap(model, Z, 3)) == Array{typeof(model),1}
end

