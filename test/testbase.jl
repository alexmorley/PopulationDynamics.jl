using PopulationDynamics
using Base.Test

### Load ### 

bsnm = "test"
datadir = abspath("test")
metadata = EPhys.getmetadata(bsnm, datadir)
sessions = [1,2];

spiketimes = loadspikes(SpikeTimes, metadata, sessions)
Z = binZ(spiketimes)

pcaica_model = PCAICA()
fit!(pcaica_model, Z)
@test typeof(weights(pcaica_model)) == Array{Float64,2}
