repeatedly(f) = (f() for _ in Iterators.repeated(nothing))

repeatedly(f, n::Integer) = (f() for _ = 1:n)

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

set!(b::Settable{T}, v::T) where {T} = (b.v = v)

Base.IteratorSize(::Type{<:Settable}) = Base.IsInfinite()

Base.iterate(b::Settable, state = nothing) = b.v, nothing

function is_logstep(it::Integer, base = 10)
    scale = floor(Int, log(base, it))
    step = base^scale
    return mod(it, step) == 0
end
