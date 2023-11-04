module ProtoGrad

_training_mode = false

is_training_mode() = _training_mode

function within_training_mode(f)
    global _training_mode
    _training_mode = true
    ret = f()
    _training_mode = false
    return ret
end

include("itertools.jl")
include("losses.jl")
include("model.jl")
include("layers.jl")
include("gradient.jl")
include("optimizers.jl")

end # module
