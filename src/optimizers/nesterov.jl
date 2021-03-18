struct NesterovIterable{T, F, S}
    w0::T
    f::F
    stepsize::S
end

function NesterovIterable(w0, f; stepsize)
    if typeof(stepsize) <: Number
        stepsize = Iterators.repeated(stepsize)
    end
    return NesterovIterable(w0, f, stepsize)
end

Base.IteratorSize(::Type{<:NesterovIterable}) = Base.IsInfinite()

mutable struct NesterovState{T, G, S}
    w::T
    f_w
    grad_f_w::G
    stepsize_iterator::S
    z::T
    z_prev::T
    theta
end

function Base.iterate(iter::NesterovIterable)
    w = copy(iter.w0)
    grad_f_w, f_w = gradient(iter.f, w)
    state = NesterovState(
        w, f_w, grad_f_w, Iterators.Stateful(iter.stepsize),
        copy(w), zero(w), 1.0
    )
    return state.w, state
end

function Base.iterate(iter::NesterovIterable, state::NesterovState)
    stepsize = popfirst!(state.stepsize_iterator)
    theta1 = (1.0 + sqrt(1.0 + 4*state.theta^2))/2.0
    extr = (state.theta - 1.0)/theta1
    state.theta = theta1
    state.w .= state.z .+ extr .* (state.z .- state.z_prev)
    state.z_prev, state.z = state.z, state.z_prev
    state.grad_f_w, state.f_w = gradient(iter.f, state.w)
    state.z .= state.w .- stepsize .* state.grad_f_w
    return state.w, state
end

struct Nesterov
    kwargs
    Nesterov(; kwargs...) = new(kwargs)
end

(n::Nesterov)(args...) = NesterovIterable(args...; n.kwargs...)
