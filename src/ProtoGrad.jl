module ProtoGrad

# this allows to write `for _ in forever` e.g. in generator expressions
const forever = Iterators.repeated(nothing)

include("losses.jl")
include("model.jl")
include("layers.jl")
include("objectives.jl")
include("gradient.jl")
include("broadcast.jl")
include("optimizers/gradient_descent.jl")
include("optimizers/nesterov.jl")
include("optimizers/barzilai_borwein.jl")
include("optimizers/adam.jl")

end # module
