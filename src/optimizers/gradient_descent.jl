struct GradientDescentIterable{T, F, S}
    w0::T
    f::F
    stepsize::S
end

function GradientDescentIterable(w0, f; stepsize)
    if typeof(stepsize) <: Number
        stepsize = Iterators.repeated(stepsize)
    end
    return GradientDescentIterable(w0, f, stepsize)
end

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
    state = GradientDescentState(
        w, f_w, grad_f_w, Iterators.Stateful(iter.stepsize)
    )
    return IterationOutput(state.w, state.f_w, state.grad_f_w), state
end

function Base.iterate(iter::GradientDescentIterable, state::GradientDescentState)
    stepsize = popfirst!(state.stepsize_iterator)
    state.w .-= stepsize .* state.grad_f_w
    state.grad_f_w, state.f_w = gradient(iter.f, state.w)
    return IterationOutput(state.w, state.f_w, state.grad_f_w), state
end

struct GradientDescent
    kwargs
    GradientDescent(; kwargs...) = new(kwargs)
end

(gd::GradientDescent)(args...) = GradientDescentIterable(args...; gd.kwargs...)
