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
    marchenko_thresh(n,B)
Get threshold for eigenvalue distribution using Marchenko-Pasur Law. See wiki: https://en.wikipedia.org/wiki/Marchenko%E2%80%93Pastur_distribution
"""
marchenko_thresh(n,B) = ((1-sqrt(n/B))^2,(1+sqrt(n/B))^2)
