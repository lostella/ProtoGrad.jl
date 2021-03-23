using NNlib: conv, maxpool, meanpool, relu, softmax

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
