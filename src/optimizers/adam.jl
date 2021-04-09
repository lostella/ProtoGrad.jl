struct AdamIterable{T, F, S, R}
    w0::T
    f::F
    stepsize::S
    beta1::R
    beta2::R
    epsilon::R
end

AdamIterable(w0, f; stepsize=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8) = AdamIterable(w0, f, to_iterator(stepsize), beta1, beta2, epsilon)

Base.IteratorSize(::Type{<:AdamIterable}) = Base.IsInfinite()

mutable struct AdamState{T, G, S, R}
    w::T
    f_w
    grad_f_w::G
    stepsize_iterator::S
    m::T
    v::T
    beta1t::R
    beta2t::R
end

function Base.iterate(iter::AdamIterable)
    w = copy(iter.w0)
    grad_f_w, f_w = gradient(iter.f, w)
    state = AdamState(
        w, f_w, grad_f_w, Iterators.Stateful(iter.stepsize),
        zero(w), zero(w), iter.beta1, iter.beta2
    )
    return IterationOutput(state.w, state.f_w, state.grad_f_w), state
end

function Base.iterate(iter::AdamIterable, state::AdamState)
    state.m .= iter.beta1 .* state.m .+ (1 - iter.beta1) .* state.grad_f_w
    state.v .= iter.beta2 .* state.v .+ (1 - iter.beta2) .* state.grad_f_w .^ 2
    stepsize = popfirst!(state.stepsize_iterator)
    alpha_t = stepsize * sqrt(1 - state.beta2t)/(1 - state.beta1t)
    state.w .-= alpha_t .* state.m ./ (sqrt.(state.v) .+ iter.epsilon)
    state.beta1t *= iter.beta1
    state.beta2t *= iter.beta2
    state.grad_f_w, state.f_w = gradient(iter.f, state.w)
    return IterationOutput(state.w, state.f_w, state.grad_f_w), state
end

struct Adam
    kwargs
    Adam(; kwargs...) = new(kwargs)
end

(a::Adam)(args...) = AdamIterable(args...; a.kwargs...)
