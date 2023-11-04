eval_with_pullback(_, _, ::Val{T}) where {T} = error("unknown AD backend $T")
eval_with_pullback(f, x, backend::Symbol) = eval_with_pullback(f, x, Val(backend))

reconstruct(m::T, ::T) where {T<:Model} = m
function reconstruct(nt::NamedTuple, m::T) where {T<:Model}
    return T((reconstruct(getfield(nt, k), getfield(m, k)) for k in fieldnames(T))...)
end
reconstruct(::Nothing, ::T) where {T<:Model} = T()
reconstruct(::Nothing, f::Function) = f
reconstruct(a::AbstractArray{T,N}, ::AbstractArray{T,N}) where {T<:Number,N} = a
function reconstruct(a::Tuple, b::Tuple)
    return tuple((reconstruct(el_a, el_b) for (el_a, el_b) in zip(a, b))...)
end
