struct NesterovMomentumSequence end

function Base.iterate(::NesterovMomentumSequence, t = 1.0)
    t_next = (1 + sqrt(1 + 4 * t^2)) / 2
    return (t - 1) / t_next, t_next
end

Base.IteratorSize(::Type{<:NesterovMomentumSequence}) = Base.IsInfinite()

@kwdef struct NesterovMethod{S,M}
    stepsize::S
    momentum_sequence::M = NesterovMomentumSequence()
end

struct NesterovState{S,V,M}
    optimizer::NesterovMethod{S}
    stepsize::S
    variable::V
    variable_prev::V
    momentum_iterator::M
end

function init(optimizer::NesterovMethod, variable)
    return NesterovState(
        optimizer,
        optimizer.stepsize,
        variable,
        zero(variable),
        Iterators.Stateful(optimizer.momentum_sequence),
    )
end

function step!(state::NesterovState, gradient)
    state.variable_prev .= state.variable
    state.variable .-= state.stepsize .* gradient
    momentum = popfirst!(state.momentum_iterator)
    return state.variable .=
        state.variable .+ momentum .* (state.variable .- state.variable_prev)
end
