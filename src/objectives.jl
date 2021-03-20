import Base: show

using LinearAlgebra

mutable struct SupervisedObjective{L, D, I, S}
    loss::L
    data_iterable::D
    next_instance::I
    next_state::S
    function SupervisedObjective(loss::L, data_iterable::D) where {L, D}
        next = iterate(data_iterable)
        next !== nothing || error("data iterable is empty")
        instance, state = next
        new{L, D, typeof(instance), typeof(state)}(loss, data_iterable, instance, state)
    end
end

show(io::IO, mo::T) where T <: SupervisedObjective = print(io, "$T($(mo.loss), $(mo.data_iterable))")

function get_instance_and_update!(f::SupervisedObjective)
    instance, state = f.next_instance, f.next_state
    next = iterate(f.data_iterable, f.next_state)
    next !== nothing || error("data iterable is exhausted")
    f.next_instance, f.next_state = next
    return instance
end

function (f::SupervisedObjective)(m)
    x, y = get_instance_and_update!(f)
    return f.loss(m(x), y)
end

# NOTE we need to explicitly implement this because Zygote doesn't like mutation

function gradient(f::SupervisedObjective, m::Model)
    x, y = get_instance_and_update!(f)
    raw_grad, out = fallback_gradient(m -> f.loss(m(x), y), m)
    grad = reconstruct(raw_grad, m)
    return grad, out
end
