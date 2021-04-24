# this allows writing `forever() do ... end`
forever(f) = (f() for _ in Iterators.repeated(nothing))

function last(itr)
    res = iterate(itr)
    if res === nothing
        throw(ArgumentError("iterator must be non-empty"))
    end
    output, state = res
    for elem in Iterators.rest(itr, state)
        output = elem
    end
    return output
end

mutable struct Settable{T}
    v::T
end

set!(b::Settable{T}, v::T) where T = (b.v = v)

Base.IteratorSize(::Type{<:Settable}) = Base.IsInfinite()

Base.iterate(b::Settable, state=nothing) = b.v, nothing
