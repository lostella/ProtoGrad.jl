struct GradientDescentIterable{T, F, S}
    w0::T
    f::F
    stepsize::S
end

GradientDescentIterable(w0, f; stepsize) = GradientDescentIterable(w0, f, stepsize)

Base.IteratorSize(::Type{<:GradientDescentIterable}) = Base.IsInfinite()

mutable struct GradientDescentState{T, R, G, S}
    w::T
    f_w::R
    grad_f_w::G
    stepsize_iterator::S
end

function Base.iterate(iter::GradientDescentIterable)
    w = copy(iter.w0)
    grad_f_w, f_w = gradient(iter.f, w)
    state = GradientDescentState(w, f_w, grad_f_w, iter.stepsize |> to_iterator |> Iterators.Stateful)
    return IterationOutput(state.w, state.f_w, state.grad_f_w), state
end

function Base.iterate(iter::GradientDescentIterable, state::GradientDescentState)
    stepsize = popfirst!(state.stepsize_iterator)
    state.w .-= stepsize .* state.grad_f_w
    state.grad_f_w, state.f_w = gradient(iter.f, state.w)
    return IterationOutput(state.w, state.f_w, state.grad_f_w), state
end

GradientDescent(; kwargs...) = IterativeAlgorithm(GradientDescentIterable; kwargs...)
