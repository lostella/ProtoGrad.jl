using LinearAlgebra
using Zygote: pullback

abstract type Model end

allfields(m::Model) = [getfield(m, k) for k in fieldnames(typeof(m))]

_push_allparams!(c::Nothing, a::AbstractArray{T}) where T <: Number = AbstractArray{T}[a]
_push_allparams!(c::Vector, a::AbstractArray{T}) where T <: Number = push!(c, a)
_push_allparams!(c, ::Function) = c
_push_allparams!(c, t::Tuple) = begin
    for el in t
        c = _push_allparams!(c, el)
    end
    return c
end
_push_allparams!(c, m::Model) = begin
    for f in allfields(m)
        c = _push_allparams!(c, f)
    end
    return c
end
allparams(m::Model) = _push_allparams!(nothing, m)

overlap(m1::Model, m2::Model) = [
    p1 for (p1, p2) in zip(ProtoGrad.allparams(m1), ProtoGrad.allparams(m2))
    if objectid(p1) == objectid(p2)
]

Base.:(==)(m1::T, m2::T) where T <: Model = all(p1 == p2 for (p1, p2) in zip(allfields(m1), allfields(m2)))
Base.:(==)(m1::T1, m2::T2) where {T1 <: Model, T2 <: Model} = false

_copy(f::Function) = f
_copy(a::AbstractArray) = copy(a)
_copy(t::Tuple) = _copy.(t)
_copy(m::T) where T <: Model = T((_copy(f) for f in allfields(m))...)
Base.copy(m::Model) = _copy(m)

_similar(f::Function) = f
_similar(a::AbstractArray) = similar(a)
_similar(t::Tuple) = _similar.(t)
_similar(m::T) where T <: Model = T((_similar(f) for f in allfields(m))...)
Base.similar(m::Model) = _similar(m)

Base.vec(m::Model) = vcat((vec(p) for p in allparams(m))...)

_zero(f::Function) = f
_zero(a::AbstractArray) = zero(a)
_zero(t::Tuple) = _zero.(t)
_zero(m::T) where T <: Model = T((_zero(f) for f in allfields(m))...)
Base.zero(m::Model) = _zero(m)

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
