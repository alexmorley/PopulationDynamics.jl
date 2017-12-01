"""
    reorder!(X,V)
Reorder Array `X` according to its correlation with `V`. Uses Hungarian
Algo to maximise diagonal of correlation matrix.
"""
function reorder!(X::Array{T,2},V::Array{T,2}) where T
    cormat = cor(V, X)
    sur_perm = munkres(-cormat)
    X .= X[:,sur_perm]
end

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
     track_partial(Z::Array{Float64,2}, V::Array{Float64,2}, masks)
As `track` but with mask on the outer produce of `V` (`Pk`). Useful for assessing contribution of specific sets of neurons,
or specific interaction between sets of neurons.
"""
function track_partial(Z::Array{Float64,2},
        V::Array{Float64,2}, masks)
    npatterns = size(V, 2)
    R = zeros(size(Z,1), npatterns, length(masks))
    Pk = zeros(size(V,1), size(V,1))
    Z = Z'
    for k in 1:npatterns
        for (mi,m) in enumerate(masks)
            Pk .= V[:,k]*V[:,k]'
            Pk[diagind(Pk)] .= 0.;
            Pk .*= m
            trackK!(view(R,:,k, mi), Pk, Z)
        end
    end
    return R
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
    f(_) = begin
        m_i = deepcopy(model)
        sample_inds .= sample(1:nsamples, nsamples, replace=true)
        fit!(m_i, Z[to_indices(Z,Tuple(x != dim ? Colon() : sample_inds for x in 1:N))...])
        return m_i
    end
    return pmap(f,1:n)
end

"""
    function confint{T<:PopulationModel}(f::Function,
                                         model::T,
                                         models::Array{T,1};
                                         α=0.05)
Get confidence interval of parameter from sample of `models` where the
parameter is defined by `f(model)` 
"""
function confint{T<:PopulationModel}(f::Function,
                                     model::T,
                                     models::Array{T,1};
                                     α=0.05)
    params = f.(models)
    reorder!.(params, [f(model)])
    lo = zeros(size(params[1])...)
    hi = copy(lo)
    α2 = (α/2)*100
    for i in eachindex(params[1])
        y = cat(1,[x[i] for x in params])
        lo[i] = StatsBase.percentile(y, α2)
        hi[i] = StatsBase.percentile(y, 100-α2)
    end
    return lo,hi
end

function confint{T<:PopulationModel}(model::T, models::Array{T,1}; kwargs...)
    confint(weights, model, models; kwargs)
end


stability(f::Function, model, models) = mean(similarity(f, model, models),2 )

"""
	similarity(f::Function=weights, model, models)
Compare a model to other models using the function `f` (defaults to weights).
"""
function similarity(f::Function, model, models)
    params = f.(models)
    orig = f(model)
    PopulationDynamics.reorder!.(params, [orig])
    cors = VectorOfArray(diag.(cor.([orig], params)))
    return cors
end

similarity(model,models) = similarity(weights, model, models)
