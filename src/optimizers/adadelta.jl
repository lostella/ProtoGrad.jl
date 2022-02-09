struct AdaDeltaIterable{T, F, R}
    w0::T
    f::F
    alpha::R
    epsilon::R
end

AdaDeltaIterable(w0, f; alpha=0.99, epsilon=1e-8) = AdaDeltaIterable(w0, f, alpha, epsilon)

Base.IteratorSize(::Type{<:AdaDeltaIterable}) = Base.IsInfinite()

mutable struct AdaDeltaState{T, G}
    w::T
    f_w
    grad_f_w::G
    update::G
    mean_squared_grad::G
    mean_squared_update::G
end

function Base.iterate(iter::AdaDeltaIterable)
    w = copy(iter.w0)
    grad_f_w, f_w = gradient(iter.f, w)
    state = AdaDeltaState(w, f_w, grad_f_w, zero(w), zero(w), zero(w))
    return IterationOutput(state.w, state.f_w, state.grad_f_w), state
end

function Base.iterate(iter::AdaDeltaIterable, state::AdaDeltaState)
    state.mean_squared_grad .= (1 - iter.alpha) .* (state.grad_f_w .^ 2) .+ iter.alpha .* state.mean_squared_grad
    state.update = sqrt.(state.mean_squared_update .+ iter.epsilon) ./ sqrt.(state.mean_squared_grad .+ iter.epsilon) .* state.grad_f_w
    state.mean_squared_update .= (1 - iter.alpha) .* (state.update .^ 2) .+ iter.alpha .* state.mean_squared_update
    state.w .-= state.update
    state.grad_f_w, state.f_w = gradient(iter.f, state.w)
    return IterationOutput(state.w, state.f_w, state.grad_f_w), state
end

AdaDelta(; kwargs...) = IterativeAlgorithm(AdaDeltaIterable; kwargs...)
