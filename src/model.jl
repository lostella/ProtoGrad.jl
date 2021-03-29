using LinearAlgebra
using Zygote: pullback

abstract type Model end

allfields(m::Model) = [getfield(m, k) for k in fieldnames(typeof(m))]

_push_allparams!(c, ::Function) = c
_push_allparams!(c, a::AbstractArray) = begin push!(c, a); c end
_push_allparams!(c, t::Tuple) = begin
    for el in t
        _push_allparams!(c, el)
    end
    return c
end
_push_allparams!(c, m::Model) = begin
    for f in allfields(m)
        _push_allparams!(c, f)
    end
    return c
end
allparams(m::Model) = _push_allparams!([], m)

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

_dot(::F, ::F) where F <: Function = 0
_dot(a1::AbstractArray, a2::AbstractArray) = dot(a1, a2)
_dot(m1::T, m2::T) where T <: Model = begin
    fields_m1 = allfields(m1)
    return length(fields_m1) == 0 ? 0 : sum((_dot(a1, a2) for (a1, a2) in zip(fields_m1, allfields(m2))))
end
_dot(t1::Tuple, t2::Tuple) = sum(_dot.(t1, t2))
LinearAlgebra.dot(m1::T, m2::T) where T <: Model = _dot(m1, m2)

Base.show(io::IO, m::T) where T <: Model = print(io, T)
