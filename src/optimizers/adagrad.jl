struct AdaGradIterable{T, F, S, R}
    w0::T
    f::F
    stepsize::S
    epsilon::R
end

AdaGradIterable(w0, f; stepsize=1e-3, epsilon=1e-8) = AdaGradIterable(w0, f, to_iterator(stepsize), epsilon)

Base.IteratorSize(::Type{<:AdaGradIterable}) = Base.IsInfinite()

mutable struct AdaGradState{T, G, S}
    w::T
    f_w
    grad_f_w::G
    stepsize_iterator::S
    sum_squared_grad::G
end

function Base.iterate(iter::AdaGradIterable)
    w = copy(iter.w0)
    grad_f_w, f_w = gradient(iter.f, w)
    state = AdaGradState(w, f_w, grad_f_w, Iterators.Stateful(iter.stepsize), zero(w))
    return IterationOutput(state.w, state.f_w, state.grad_f_w), state
end

function Base.iterate(iter::AdaGradIterable, state::AdaGradState)
    state.sum_squared_grad .+= state.grad_f_w .^ 2
    stepsize = popfirst!(state.stepsize_iterator)
    state.w .-= stepsize .* state.grad_f_w ./ (sqrt.(state.sum_squared_grad) .+ iter.epsilon)
    state.grad_f_w, state.f_w = gradient(iter.f, state.w)
    return IterationOutput(state.w, state.f_w, state.grad_f_w), state
end

struct AdaGrad
    kwargs
    AdaGrad(; kwargs...) = new(kwargs)
end

(a::AdaGrad)(args...) = AdaGradIterable(args...; a.kwargs...)
