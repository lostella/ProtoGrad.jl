using Zygote: pullback

function fallback_gradient(f, x)
    out, pb = pullback(f, x)
    grad = pb(one(out))[1]
    return grad, out
end

gradient(f, x) = fallback_gradient(f, x)

# TODO define this using ChainRulesCore.rrule

function gradient(f, m::Model)
    raw_grad, out = fallback_gradient(f, m)
    grad = reconstruct(raw_grad, m)
    return grad, out
end

reconstruct(m::T, ::T) where T <: Model = m
reconstruct(z::Zero{T}, ::Frozen{T}) where T <: Model = z
reconstruct(nt::NamedTuple, m::T) where T <: Model = T((reconstruct(getfield(nt, k), getfield(m, k)) for k in fieldnames(T))...)
reconstruct(::Nothing, ::T) where T <: Model = T()
reconstruct(::Nothing, f::Function) = f
reconstruct(a::AbstractArray{T, N}, ::AbstractArray{T, N}) where {T <: Number, N} = a
reconstruct(a::Tuple, b::Tuple) = tuple((reconstruct(el_a, el_b) for (el_a, el_b) in zip(a, b))...)
