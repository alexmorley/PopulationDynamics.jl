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
