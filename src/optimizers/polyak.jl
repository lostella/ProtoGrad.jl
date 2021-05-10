struct PolyakIterable{T, F, S, M}
    w0::T
    f::F
    stepsize::S
    momentum::M
end

PolyakIterable(w0, f; stepsize, momentum) = PolyakIterable(w0, f, to_iterator(stepsize), to_iterator(momentum))

Base.IteratorSize(::Type{<:PolyakIterable}) = Base.IsInfinite()

mutable struct PolyakState{T, R, G, S, M}
    w::T
    f_w::R
    grad_f_w::G
    stepsize_iterator::S
    momentum_iterator::M
    w_prev::T
end

function Base.iterate(iter::PolyakIterable)
    w = copy(iter.w0)
    grad_f_w, f_w = gradient(iter.f, w)
    state = PolyakState(w, f_w, grad_f_w, Iterators.Stateful(iter.stepsize), Iterators.Stateful(iter.momentum), copy(w))
    return IterationOutput(state.w, state.f_w, state.grad_f_w), state
end

function Base.iterate(iter::PolyakIterable, state::PolyakState)
    stepsize = popfirst!(state.stepsize_iterator)
    momentum = popfirst!(state.momentum_iterator)
    state.w_prev .= state.w .- stepsize .* state.grad_f_w .+ momentum .* (state.w - state.w_prev)
    state.w, state.w_prev = state.w_prev, state.w
    state.grad_f_w, state.f_w = gradient(iter.f, state.w)
    return IterationOutput(state.w, state.f_w, state.grad_f_w), state
end

struct Polyak
    kwargs
    Polyak(; kwargs...) = new(kwargs)
end

(alg::Polyak)(args...) = PolyakIterable(args...; alg.kwargs...)
