struct NesterovMomentum end

function Base.iterate(::NesterovMomentum, t=1.0)
    t_next = (1 + sqrt(1 + 4 * t^2)) / 2
    return (t - 1) / t_next, t_next
end

struct NesterovIterable{T, F, S, M}
    w0::T
    f::F
    stepsize::S
    momentum::M
end

NesterovIterable(w0, f; stepsize, momentum=NesterovMomentum()) = NesterovIterable(w0, f, stepsize, momentum)

Base.IteratorSize(::Type{<:NesterovIterable}) = Base.IsInfinite()

mutable struct NesterovState{T, G, S, M}
    w::T
    f_w
    grad_f_w::G
    stepsize_iterator::S
    momentum_iterator::M
    z::T
    z_prev::T
end

function Base.iterate(iter::NesterovIterable)
    w = copy(iter.w0)
    grad_f_w, f_w = gradient(iter.f, w)
    state = NesterovState(
        w, f_w, grad_f_w,
        iter.stepsize |> to_iterator |> Iterators.Stateful,
        iter.momentum |> to_iterator |> Iterators.Stateful,
        copy(w), zero(w)
    )
    return IterationOutput(state.w, state.f_w, state.grad_f_w), state
end

function Base.iterate(iter::NesterovIterable, state::NesterovState)
    stepsize = popfirst!(state.stepsize_iterator)
    state.z .= state.w .- stepsize .* state.grad_f_w
    momentum = popfirst!(state.momentum_iterator)
    state.w .= state.z .+ momentum .* (state.z .- state.z_prev)
    state.z_prev, state.z = state.z, state.z_prev
    state.grad_f_w, state.f_w = gradient(iter.f, state.w)
    return IterationOutput(state.w, state.f_w, state.grad_f_w), state
end

Nesterov(; kwargs...) = IterativeAlgorithm(NesterovIterable; kwargs...)
