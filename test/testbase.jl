using StatsBase
using Base.Test

### Load ### 

bsnm = "test"
datadir = abspath("test")
metadata = EPhys.getmetadata(bsnm, datadir)
sessions = [1,2];


