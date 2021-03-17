module ProtoGrad

include("losses.jl")
include("model.jl")
include("layers.jl")
include("objectives.jl")
include("autodiff.jl")
include("broadcast.jl")
include("itertools.jl")
include("optimizers/gradient_descent.jl")
include("optimizers/nesterov.jl")
include("optimizers/barzilai_borwein.jl")
include("optimizers/adam.jl")

end # module
