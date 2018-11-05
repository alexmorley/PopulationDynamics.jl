export PopulationModel

abstract type PopulationModel end

function track(model::PopulationModel, Z::Array{T,2}, M::BitArray{2}) where T
    track(Z, weights(model), M)
end

# all subtypes of PopulationModel must have at least two methods
# PopulationModel(opts...)  # a constructor
# fit(p::Type{PopulationModel}, Z::Array{T,N), opts) # fit the model to an input


# For "spatial" models we might additionally have
# track(p::PopulationModel, Z)      # track the expression of populations over time
# bootstrap(p::PopulationModel, Z, fitopts)  # recalulate the model using a resampled Z 
#                                   (applicable where Z is a 2-order tensor)


# ^ these methods should also work with the basic dimensionality reduction techniques in MultivariateStats

# Where Z is a 3rd or higher order tensor track & bootstrap will have to take a dimension 
