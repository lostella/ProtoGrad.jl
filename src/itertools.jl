const forever = Iterators.repeated(nothing)

struct ExponentialDecay{T}
    v0::T
    alpha::T
end

Base.IteratorSize(::Type{ExponentialDecay{T}}) where T = Base.IsInfinite()
Base.IteratorEltype(::Type{ExponentialDecay{T}}) where T = T

function Base.iterate(iter::ExponentialDecay, state=iter.v0)
    return state, iter.alpha*state
end

struct BoundedExponentialDecay{T}
    v0::T
    alpha::T
    low::T
end

Base.IteratorSize(::Type{BoundedExponentialDecay{T}}) where T = Base.IsInfinite()
Base.IteratorEltype(::Type{BoundedExponentialDecay{T}}) where T = T

function Base.iterate(iter::BoundedExponentialDecay, state=iter.v0)
    next_state = iter.alpha*state
    return state, next_state >= iter.low ? next_state : state
end

struct HaltingIterable{I, F}
    iter::I
    fun::F
end

Base.IteratorSize(::Type{HaltingIterable{I, F}}) where {I, F} = Base.SizeUnknown()
Base.IteratorEltype(::Type{HaltingIterable{I, F}}) where {I, F} = Base.IteratorEltype(I)

Base.eltype(iter::HaltingIterable{I, F}) where {I, F} = eltype(iter.iter)

function Base.iterate(iter::HaltingIterable)
    next = iterate(iter.iter)
    return dispatch(iter, next)
end

function Base.iterate(iter::HaltingIterable, (instruction, state))
    if instruction == :halt return nothing end
    next = iterate(iter.iter, state)
    return dispatch(iter, next)
end

function dispatch(iter::HaltingIterable, next)
    if next === nothing return nothing end
    return next[1], (iter.fun(next[1]) ? :halt : :continue, next[2])
end

halt(iter::I, fun::F) where {I, F} = HaltingIterable{I, F}(iter, fun)

struct MinSoFar{I}
    iter::I
end

Base.IteratorSize(::Type{MinSoFar{I}}) where I = Base.IteratorSize(I)
Base.IteratorEltype(::Type{MinSoFar{I}}) where I = Base.IteratorEltype(I)

Base.length(iter::MinSoFar{I}) where I = length(iter.iter)
Base.axes(iter::MinSoFar{I}) where I = axes(iter.iter)

function Base.iterate(iter::MinSoFar, (iterator, min_so_far)=(Iterators.Stateful(iter.iter), nothing))
    if isempty(iterator)
        return nothing
    end
    v = popfirst!(iterator)
    min_so_far = min_so_far === nothing || v < min_so_far ? v : min_so_far
    return min_so_far, (iterator, min_so_far)
end

struct StateInspector{I}
    iter::I
end

inspect(iter) = StateInspector{typeof(iter)}(iter)

function Base.iterate(iter::StateInspector, state...)
    res = iterate(iter.iter, state...)
    if res === Nothing
        return Nothing
    else
        return res[2], res[2]
    end
end
