using Base: RefValue
using Base.Broadcast: Style, DefaultArrayStyle, Broadcasted, flatten

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
find_model(::Tuple{}) = nothing
find_model(m::Model, rest) = m
find_model(::Any, rest) = find_model(rest)

function Base.copy(bc::Broadcasted{Style{T}}) where T <: Model
    m = find_model(bc)
    dest = similar(m)
    copyto!(dest, bc)
end

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
