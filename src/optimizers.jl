to_iterator(x::Any) = x
to_iterator(n::T) where {T<:Number} = Iterators.repeated(n)

include("optimizers/gradient_descent.jl")
include("optimizers/heavy_ball_method.jl")
include("optimizers/nesterov_method.jl")
include("optimizers/adam.jl")
