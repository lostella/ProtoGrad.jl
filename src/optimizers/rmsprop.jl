struct RMSPropIterable{T, F, S, R}
    w0::T
    f::F
    stepsize::S
    alpha::R
    epsilon::R
end

RMSPropIterable(w0, f; stepsize=1e-3, alpha=0.99, epsilon=1e-8) = RMSPropIterable(w0, f, stepsize, alpha, epsilon)

Base.IteratorSize(::Type{<:RMSPropIterable}) = Base.IsInfinite()

mutable struct RMSPropState{T, G, S}
    w::T
    f_w
    grad_f_w::G
    stepsize_iterator::S
    mean_squared_grad::T
end

function Base.iterate(iter::RMSPropIterable)
    w = copy(iter.w0)
    grad_f_w, f_w = gradient(iter.f, w)
    state = RMSPropState(w, f_w, grad_f_w, iter.stepsize |> to_iterator |> Iterators.Stateful, zero(w))
    return IterationOutput(state.w, state.f_w, state.grad_f_w), state
end

function Base.iterate(iter::RMSPropIterable, state::RMSPropState)
    state.mean_squared_grad .= (1 - iter.alpha) .* (state.grad_f_w .^ 2) .+ iter.alpha .* state.mean_squared_grad
    stepsize = popfirst!(state.stepsize_iterator)
    state.w .-= stepsize .* state.grad_f_w ./ (sqrt.(state.mean_squared_grad) .+ iter.epsilon)
    state.grad_f_w, state.f_w = gradient(iter.f, state.w)
    return IterationOutput(state.w, state.f_w, state.grad_f_w), state
end

RMSProp(; kwargs...) = IterativeAlgorithm(RMSPropIterable; kwargs...)
