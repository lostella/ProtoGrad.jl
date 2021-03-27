struct IterationOutput{M, R}
    model::M
    value::R
    gradient::M
end

include("optimizers/gradient_descent.jl")
include("optimizers/nesterov.jl")
include("optimizers/barzilai_borwein.jl")
include("optimizers/adam.jl")
