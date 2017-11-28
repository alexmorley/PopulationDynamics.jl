@everywhere using PopulationDynamics
@everywhere using Base.Test

### Load ### 

bsnm = "test"
datadir = abspath("test")
metadata = EPhys.getmetadata(bsnm, datadir)
sessions = [2];

spiketimes = loadspikes(SpikeTimes, metadata, sessions)
spiketimes = filter(x->length(x)>100, spiketimes)
Z = binZ(spiketimes)'
t = loadtrigtimes("toneSucrose_pulse", metadata, sessions)
Z3 = binZ3(spiketimes,t)

### Test ###
function testmodel(model,name, Z)
    println("   $name")
    println("       fit")
    fit!(model, Z)
    @test typeof(weights(model)) == Array{Float64,2}
    println("       track")
    @test track(model, zeros(size(Z,1),100)) == zeros(model.k,100)
    println("       bootstrap")
    btstrp = bootstrap(model, Z, 3)
    @test typeof(btstrp) == Array{typeof(model),1}
    println("       confint")
    lo, hi = confint(weights, model, btstrp)
    @test sum(hi) > sum(lo)
end

println("Testing 2nd Order Models...")
modelnames = ["PCA",
              "PCAICA"]
models = [PCAICA(),
          PCA()]
testmodel.(models, modelnames, [Z])

println("")
println("Testing 3rd Order Models...")
modelnames = ["Canonical Polyadic Decomposition (CPD)"]
models = [CP(2)]
testmodel.(models, modelnames, [Z3])
