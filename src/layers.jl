using NNlib: conv, maxpool, meanpool, relu, softmax
using ChainRulesCore

# Initialization utils

xavier_uniform(T::Type, dims...; fan_in, fan_out) = sqrt(T(6) / (fan_in + fan_out)) * (2 * rand(T, dims...) .- 1)
xavier_uniform(dims...; fan_in, fan_out) = xavier_uniform(Float32, dims..., fan_in=fan_in, fan_out=fan_out)

# Linear layers

struct Linear{TW, Tb} <: Model
    W::TW
    b::Tb
end

function Linear(T::Type, dims::Pair)
    W = xavier_uniform(T, dims[2], dims[1], fan_in=dims[1], fan_out=dims[2])
    b = zeros(T, dims[2])
    return Linear(W, b)
end

Linear(dims::Pair) = Linear(Float32, dims)

function (l::Linear)(x)
    size_x = size(x)
    x_reshaped = reshape(x, size_x[1], :)
    out_reshaped = l.W * x_reshaped .+ l.b
    out = reshape(out_reshaped, :, size_x[2:end]...)
    return out
end

# Convolutional layers

struct Conv{Tw, Tb} <: Model
    w::Tw
    b::Tb
end

function Conv(T::Type, num_channels::Pair, filter_size::Tuple)
    fan_in = prod(filter_size) * num_channels[1]
    fan_out = prod(filter_size) * num_channels[2]
    w = xavier_uniform(T, filter_size..., num_channels..., fan_in=fan_in, fan_out=fan_out)
    b = zeros(T, 1, 1, num_channels[2])
    return Conv(w, b)
end

Conv(num_channels::Pair, filter_size::Tuple) = Conv(Float32, num_channels, filter_size)

(c::Conv)(x) = conv(x, c.w) .+ c.b

# Recurrent layers

struct Unroll{C, I, S}
    cell::C
    input_sequence::I
    initial_state::S
end

Base.length(itr::Unroll) = length(itr.input_sequence)

function Base.iterate(itr::Unroll, state=(itr.input_sequence, itr.initial_state))
    remaining_inputs, cell_state = state
    p = iterate(remaining_inputs)
    if p === nothing
        return nothing
    end
    next_input, inputs_state = p
    output, cell_state = itr.cell(next_input, cell_state)
    return output, (Iterators.rest(remaining_inputs, inputs_state), cell_state)
end

struct RecurrentLayer{axis, C, S} <: Model
    cell::C
    state_initializer::S
end

RecurrentLayer(cell::C, state_initializer::S; axis) where {C, S} = RecurrentLayer{axis, C, S}(cell, state_initializer)

_split_view_indexes(k, axis, ndims) = tuple((i == axis ? k : Colon() for i in 1:ndims)...)

split(x, axis) = [view(x, _split_view_indexes(k, axis, ndims(x))...) for k in 1:size(x, axis)]

_stack_reshape_size(axis, sz) = tuple((i == axis ? 1 : (i < axis ? sz[i] : sz[i-1]) for i in 1:length(sz)+1)...)

stack(xs, axis) = cat((reshape(x, _stack_reshape_size(axis, size(xs[1]))) for x in xs)..., dims=axis)

function (m::RecurrentLayer{axis})(x) where axis
    xs = split(x, axis)
    outputs = collect(Unroll(m.cell, xs, m.state_initializer()))
    return stack(outputs, axis)
end

struct RNNCell{M, V, A} <: Model
    Wi::M
    Wh::M
    b::V
    act::A
end

function (cell::RNNCell)(x, h)
    size_x = size(x)
    x_reshaped = reshape(x, size_x[1], :)
    h = cell.act.(cell.Wi * x_reshaped .+ cell.Wh * h .+ cell.b)
    return reshape(h, :, size_x[2:end]...), h
end

function RNNCell(T::Type, dims::Pair, act)
    Wi = xavier_uniform(T, dims[2], dims[1], fan_in=dims[1], fan_out=dims[2])
    Wh = xavier_uniform(T, dims[2], dims[2], fan_in=dims[2], fan_out=dims[2])
    b = zeros(T, dims[2])
    return RNNCell(Wi, Wh, b, act)
end

RNNCell(args...; kwargs...) = RNNCell(Float32, args...; kwargs...)

RNN(T::Type, dims::Pair, act; axis=2) = RecurrentLayer(RNNCell(T, dims, act), () -> zeros(T, dims[2]), axis=axis)

RNN(args...; kwargs...) = RNN(Float32, args...; kwargs...)

# TODO *immediately* test whether gradients work OK with this

# Composition of layers

struct Compose{T <: Tuple} <: Model
    layers::T
end

Compose(first, rest...) = Compose((first, rest...))

function (m::Compose)(x)
    y = x
    for layer in m.layers
        y = layer(y)
    end
    return y
end

# Parameter-free layers (activations, pooling, reshaping...)

struct ReLU <: Model end

(::ReLU)(x) = relu.(x)

struct SoftMax{D} <: Model end

SoftMax(d=1) = SoftMax{d}()

(::SoftMax{D})(x) where D = softmax(x, dims=D)

struct MaxPool{S} <: Model end

MaxPool(s) = MaxPool{s}()

(p::MaxPool{S})(x) where S = maxpool(x, S)

struct MeanPool{S} <: Model end

MeanPool(s) = MeanPool{s}()

(p::MeanPool{S})(x) where S = meanpool(x, S)

struct Reshape{S} <: Model end

Reshape(s) = Reshape{s}()

(r::Reshape{S})(x) where S = reshape(x, S)

# Dropout

_dropout_shape(s, ::Colon) = size(s)
_dropout_shape(s, dims) = tuple((i in dims ? si : 1 for (i, si) in enumerate(size(s)))...)

function _dropout_mask(x::AbstractArray{T}, p; dims=:) where T
    y = rand(eltype(x), _dropout_shape(x, dims)...)
    y .= ifelse.(y .> p, T(1/(1 - p)), T(0))
    return y
end

function dropout(x, p; dims=:)
    if !is_training_mode()
        return x
    end
    y = _dropout_mask(x, p, dims=dims)
    return x .* y
end

function ChainRulesCore.rrule(::typeof(dropout), x, p; dims=:)
    if !is_training_mode()
        return x, c -> (ChainRulesCore.NO_FIELDS, c, ChainRulesCore.Zero())
    end
    y = _dropout_mask(x, p, dims=dims)
    return x .* y, c -> (ChainRulesCore.NO_FIELDS, c .* y, ChainRulesCore.Zero())
end

struct Dropout{P, D} <: Model end

Dropout(p, dims=:) = Dropout{p, dims}()

(d::Dropout{P, D})(x) where {P, D} = dropout(x, P, dims=D)
