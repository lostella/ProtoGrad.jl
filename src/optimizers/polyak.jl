struct PolyakIterable{T, F, S, M}
    w0::T
    f::F
    stepsize::S
    momentum::M
end

PolyakIterable(w0, f; stepsize, momentum) = PolyakIterable(w0, f, stepsize, momentum)

Base.IteratorSize(::Type{<:PolyakIterable}) = Base.IsInfinite()

mutable struct PolyakState{T, R, G, S, M}
    w::T
    f_w::R
    grad_f_w::G
    stepsize_iterator::S
    momentum_iterator::M
    step::T
end

function Base.iterate(iter::PolyakIterable)
    w = copy(iter.w0)
    grad_f_w, f_w = gradient(iter.f, w)
    state = PolyakState(w, f_w, grad_f_w, iter.stepsize |> to_iterator |> Iterators.Stateful, iter.momentum |> to_iterator |> Iterators.Stateful, zero(w))
    return IterationOutput(state.w, state.f_w, state.grad_f_w), state
end

function Base.iterate(iter::PolyakIterable, state::PolyakState)
    stepsize = popfirst!(state.stepsize_iterator)
    momentum = popfirst!(state.momentum_iterator)
    state.step .= momentum .* state.step .- stepsize .* state.grad_f_w
    state.w .+= state.step
    state.grad_f_w, state.f_w = gradient(iter.f, state.w)
    return IterationOutput(state.w, state.f_w, state.grad_f_w), state
end

Polyak(; kwargs...) = IterativeAlgorithm(PolyakIterable; kwargs...)
