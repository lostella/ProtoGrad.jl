using LinearAlgebra
using Zygote: pullback

abstract type Model end

map_recursively(fun, f::Function) = f
map_recursively(fun, a::AbstractArray) = fun(a)
map_recursively(fun, t::Tuple) = fun.(t)
map_recursively(fun, m::T) where T <: Model = T((map_recursively(fun, getfield(m, k)) for k in fieldnames(T))...)

for fun in (:similar, :copy, :zero)
    _fun = Symbol("_", fun)
    @eval begin
        $_fun(x) = $fun(x)
        $_fun(f::Function) = f
        Base.$fun(m::Model) = map_recursively($_fun, m)
    end
end

# NOTE: looks like this is slower than growing an array with push!
# _yield_allparams(::Function) = ()
# _yield_allparams(t::Tuple) = Iterators.flatten((_yield_allparams(el) for el in t))
# _yield_allparams(a::AbstractArray{T}) where T <: Number = (a,)
# _yield_allparams(m::T) where T <: Model = Iterators.flatten((_yield_allparams(getfield(m, k)) for k in fieldnames(T)))
# allparams(m::Model) = _yield_allparams(m)

_push_allparams!(::Nothing, a::AbstractArray{T}) where T <: Number = AbstractArray{T}[a]
_push_allparams!(c::Vector, a::AbstractArray{T}) where T <: Number = push!(c, a)
_push_allparams!(c, ::Function) = c
_push_allparams!(c, t::Tuple) = begin
    for el in t
        c = _push_allparams!(c, el)
    end
    return c
end
_push_allparams!(c, m::T) where T <: Model = begin
    for k in fieldnames(T)
        c = _push_allparams!(c, getfield(m, k))
    end
    return c
end
allparams(m::Model) = _push_allparams!(nothing, m)

overlap(m1::Model, m2::Model) = [
    p1 for (p1, p2) in zip(ProtoGrad.allparams(m1), ProtoGrad.allparams(m2))
    if objectid(p1) == objectid(p2)
]

Base.vec(m::Model) = vcat((vec(p) for p in allparams(m))...)

Base.IteratorSize(::Type{<:Model}) = Base.HasLength()
Base.length(m::Model) = sum(length(p) for p in allparams(m))

function Base.iterate(m::Model, itr=Iterators.flatten(allparams(m)))
    y = iterate(itr)
    if y === nothing
        return nothing
    end
    val, s = y
    return val, Iterators.rest(itr, s)
end

Base.show(io::IO, m::T) where T <: Model = print(io, T)

LinearAlgebra.dot(m1::T, m2::T) where T <: Model = sum((dot(p1, p2) for (p1, p2) in zip(allparams(m1), allparams(m2))))
