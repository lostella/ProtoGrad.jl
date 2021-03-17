using LinearAlgebra

struct BarzilaiBorweinIterable{T, F, R}
    w0::T
    f::F
    alpha::R
end

BarzilaiBorweinIterable(w0, f; alpha) = BarzilaiBorweinIterable(w0, f, alpha)

Base.IteratorSize(::Type{<:BarzilaiBorweinIterable}) = Base.IsInfinite()

mutable struct BarzilaiBorweinState{T, G, R}
    w::T
    f_w
    grad_f_w::G
    stepsize::R
end

function Base.iterate(iter::BarzilaiBorweinIterable)
    w = copy(iter.w0)
    grad_f_w, f_w = gradient(iter.f, w)
    state = BarzilaiBorweinState(
        w, f_w, grad_f_w, iter.alpha
    )
    return state.w, state
end

function Base.iterate(iter::BarzilaiBorweinIterable, state::BarzilaiBorweinState)
    state.w .-= state.stepsize .* state.grad_f_w
    y, state.f_w = gradient(iter.f, state.w)
    y .-= state.grad_f_w
    state.stepsize *= -dot(state.grad_f_w, state.grad_f_w) / dot(state.grad_f_w, y)
    state.grad_f_w .+= y
    return state.w, state
end

struct BarzilaiBorwein
    kwargs
    BarzilaiBorwein(; kwargs...) = new(kwargs)
end

function (bb::BarzilaiBorwein)(args...)
    BarzilaiBorweinIterable(args...; bb.kwargs...)
end
