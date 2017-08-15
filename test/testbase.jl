using PopulationDynamics
using Base.Test

### Load ### 

bsnm = "test"
datadir = abspath("test")
metadata = EPhys.getmetadata(bsnm, datadir)
sessions = [1,2];

spiketimes = loadspikes(SpikeTimes, metadata, sessions)
Z = binZ(spiketimes)'

model = PCAICA()
fit!(model, Z)
@test typeof(weights(model)) == Array{Float64,2}

@test track(model, zeros(size(Z,1),100)) == zeros(model.k,100)
