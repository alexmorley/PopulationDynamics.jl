"""
	function normweightvectors!(V)
Normalise the weight vector of each pattern so that the maximum absolute value
is one.
"""
function normweightvectors!{T}(V::Array{T,2})
    for i in 1:size(V,2)
        weightvec = view(V,:,i)
        if abs(minimum(weightvec)) > maximum(weightvec)
            weightvec .= -weightvec
        end
        weightvec ./= vecnorm(weightvec)
    end
end

"""
    function track(Z, V)
Track zscored firing rates Z using weight vectors V.
Tracking uses quadratic form
    R(t) = z(t)' * Pk(t) * (zt)
"""
function track{T<:Float64}(Z::Array{T,2}, V::Array{T,2})
    K = size(V,2)
    R = zeros(Float64, size(Z,2), K)
    for k in 1:K
        Rk = view(R,:,k)
        Pk = V[:,k]*V[:,k]'
        Pk[diagind(Pk)] .= 0.
        trackK!(Rk, Pk, Z)
    end
    return R
end

function trackK!(Rk,Pk,Z)
    for t in 1:size(Z,2)
        Rk[t] = Z[:,t]'*Pk*Z[:,t]
    end
    nothing
end

"""
    marchenko_thresh(n,B)
Get threshold for eigenvalue distribution using Marchenko-Pasur Law. See wiki: https://en.wikipedia.org/wiki/Marchenko%E2%80%93Pastur_distribution
"""
marchenko_thresh(n,B) = ((1-sqrt(n/B))^2,(1+sqrt(n/B))^2)


"""
    bootstrap(model::PopulationModel, Z, n=100, dim=2)
Fit model `n` times sampling (with replacement) from dimension `dim`.
"""
function bootstrap{T,N}(model::PopulationModel, Z::Array{T,N}, n=100, dim=2)
    nsamples = size(Z,dim)
    sample_inds = zeros(Int,nsamples)
    models = [deepcopy(model) for _ in 1:n]
    for i in 1:n
        sample_inds .= sample(1:nsamples, nsamples, replace=true)
        fit!(models[i], Z[to_indices(Z,Tuple(x != dim ? Colon() : sample_inds for x in 1:N))...])
    end
    return models
end

"""
    function confint{T<:PopulationModel}(f::Function,
                                    models::Array{T,1};
                                    α=0.05)
Get confidence interval of parameter from sample of `models` where the
parameter is defined by `f(model)` 
"""
function confint{T<:PopulationModel}(f::Function,
                                    models::Array{T,1};
                                    α=0.05)
    params = f.(models)
    lo = zeros(size(params[1])...)
    hi = copy(lo)
    α2 = α/2
    for i in eachindex(params[1])
        y = cat(1,[x[i] for x in params])
        lo[i] = StatsBase.percentile(y, α2)
        hi[i] = StatsBase.percentile(y, 100-α2)
    end
    return lo,hi
end
function confint{T<:PopulationModel}(models::Array{T,1};args...)
    confint(weights,models)
end
