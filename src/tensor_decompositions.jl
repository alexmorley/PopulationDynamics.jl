export CP

mutable struct CP <: PopulationModel
    k::Int
    cp::Union{CANDECOMP{Float64,3},NaN}
	function CP(k, cp=NaN)
		new(k,cp)
	end
end

function fit!(model::CP, Z::Array{Float64,3})
	model.cp = candecomp(Z, model.k, ([randn(x,r) for x in size(Z)]...),
    compute_error=true, method=:ALS)
end

weights(model::CP) = model.cp.factors[1]

factors(model::CP, dim=0) = dim == 0 ? model.cp.factors : model.cp.factors[dim]
