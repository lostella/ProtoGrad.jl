using Base: RefValue
using Base.Broadcast: Style, DefaultArrayStyle, Broadcasted, flatten
using LinearAlgebra
using Zygote: pullback

abstract type Model end

# Basic operations

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

# NOTE: looks like the above definition is faster than what follows
# _yield_allparams(::Function) = ()
# _yield_allparams(t::Tuple) = Iterators.flatten((_yield_allparams(el) for el in t))
# _yield_allparams(a::AbstractArray{T}) where T <: Number = (a,)
# _yield_allparams(m::T) where T <: Model = Iterators.flatten((_yield_allparams(getfield(m, k)) for k in fieldnames(T)))
# allparams(m::Model) = _yield_allparams(m)

Base.vec(m::Model) = vcat((vec(p) for p in allparams(m))...)

overlap(m1::Model, m2::Model) = [
    p1 for (p1, p2) in zip(ProtoGrad.allparams(m1), ProtoGrad.allparams(m2))
    if objectid(p1) == objectid(p2)
]

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

# Iteration

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

# Indexing

Base.firstindex(m::Model) = 1
Base.lastindex(m::Model) = length(m)

Base.getindex(m::Model, i) = begin
    # TODO optimize (no need to call allparams)
    ps = allparams(m)
    j = 1
    while i > length(ps[j])
        i -= length(ps[j])
        j += 1
    end
    return ps[j][i]
end

Base.setindex!(m::Model, v, i) = begin
    # TODO optimize (no need to call allparams)
    ps = allparams(m)
    j = 1
    while i > length(ps[j])
        i -= length(ps[j])
        j += 1
    end
    ps[j][i] = v
end

# Broadcast

Base.axes(::Model) = nothing
Base.Broadcast.broadcastable(m::Model) = m

Base.Broadcast.BroadcastStyle(::Type{T}) where T <: Model = Style{T}()
Base.Broadcast.BroadcastStyle(s::Style{T}, ::DefaultArrayStyle{0}) where T <: Model = s

Base.Broadcast.instantiate(bc::Broadcasted{Style{T}}) where T <: Model = bc

# The idea here is:
#   - have a recursive procedure that goes down to the leaves of the struct
#   - where to descend is dictated by the structure of the destination container
#   - scalars get passed down as they are
#   - once the leaves (numerical arrays or functions) are reached, just use default broadcasting

_getindex(a, idx) = getindex(a, idx)
_getindex(rv::RefValue, idx) = rv
_getindex(n::Number, idx) = n
_getindex(f::Function, idx) = f

_getfield(s, k) = getfield(s, k)
_getfield(rv::RefValue, k) = rv
_getfield(n::Number, k) = n
_getfield(f::Function, k) = f

function recursive_copyto!(a::AbstractArray, f, args)
    a .= f.(args...)
end

recursive_copyto!(f::Function, _, args) = return

function recursive_copyto!(t::Tuple, f, args)
    for (i, el) in Iterators.enumerate(t)
        args_ith = [_getindex(arg, i) for arg in args]
        recursive_copyto!(el, f, args_ith)
    end
end

function recursive_copyto!(m::T, f, args) where T <: Model
    for k in fieldnames(T)
        dest_k = getfield(m, k)
        args_k = [_getfield(arg, k) for arg in args]
        recursive_copyto!(dest_k, f, args_k)
    end
end

function Base.copyto!(dest::T, bc::Broadcasted{Style{T}}) where T <: Model
    flat_bc = flatten(bc)
    recursive_copyto!(dest, flat_bc.f, flat_bc.args)
    dest
end

find_model(bc::Base.Broadcast.Broadcasted) = find_model(bc.args)
find_model(args::Tuple) = find_model(find_model(args[1]), Base.tail(args))
find_model(x) = x
find_model(m::Model, rest) = m
find_model(::Any, rest) = find_model(rest)

function Base.copy(bc::Broadcasted{Style{T}}) where T <: Model
    m = find_model(bc)
    dest = similar(m)
    copyto!(dest, bc)
end

# Vector space operations

for op in (:+, :-)
    @eval begin
        function Base.$op(a::T, b::T) where T <: Model
            Base.broadcast($op, a, b)
        end

        function Base.$op(a::Model, b::Number)
            Base.broadcast($op, a, b)
        end

        function Base.$op(a::Number, b::Model)
            Base.broadcast($op, a, b)
        end
    end
end

for op in (:*, :/)
    @eval function Base.$op(a::Model, b::Number)
        Base.broadcast($op, a, b)
    end
end

function Base.:*(a::Number, b::Model)
    Base.broadcast(*, a, b)
end

function Base.:\(a::Number, b::Model)
    Base.broadcast(/, a, b)
end

# Linear algebra

LinearAlgebra.dot(m1::T, m2::T) where T <: Model = sum((dot(p1, p2) for (p1, p2) in zip(allparams(m1), allparams(m2))))
