using Statistics

mae(output, label; agg=mean) = agg(abs.(output .- label))

mse(output, label; agg=mean) = agg((output .- label) .^ 2)

function cross_entropy(probs, label; dims=1, agg=mean)
    ce_summands = label .* log.(probs .+ eps(eltype(probs)))
    agg(.-sum(ce_summands, dims=dims))
end

function class_error(probs, label; dims=1, agg=mean)
    class_predicted = argmax(probs, dims=dims)
    class_actual = argmax(label, dims=dims)
    return agg(1 .- (class_predicted .== class_actual))
end
