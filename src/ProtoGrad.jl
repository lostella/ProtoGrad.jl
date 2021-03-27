module ProtoGrad

# this allows to write `for _ in forever` e.g. in generator expressions
const forever = Iterators.repeated(nothing)

_training_mode = false

is_training_mode() = _training_mode

function within_training_mode(f)
    global _training_mode
    _training_mode = true
    ret = f()
    _training_mode = false
    return ret
end

include("losses.jl")
include("model.jl")
include("layers.jl")
include("objectives.jl")
include("gradient.jl")
include("broadcast.jl")
include("optimizers.jl")

end # module
