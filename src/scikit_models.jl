abstract type ScikitModel end

mutable struct FactorAnalysis
    k::Int
    model
    function FactorAnalysis(k)
        if !isdefined(:FactorAnalysis)
            using ScikitLearn
            @sk_import decomposition: FactorAnalysis
        end
        new(k, FactorAnalysis(k))
    end
end
