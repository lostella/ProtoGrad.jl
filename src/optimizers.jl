struct IterationOutput{M, R}
    solution::M
    value::R
    gradient::M
end

to_iterator(x::Any) = x
to_iterator(n::T) where T <: Number = Iterators.repeated(n)

include("optimizers/gradient_descent.jl")
include("optimizers/nesterov.jl")
include("optimizers/barzilai_borwein.jl")
include("optimizers/adagrad.jl")
include("optimizers/rmsprop.jl")
include("optimizers/adadelta.jl")
include("optimizers/adam.jl")
