using MLDatasets: MNIST
using Flux: onehotbatch
using ProtoGrad: Conv, Linear, Dropout, Compose, maxpool, relu, softmax
using ProtoGrad: cross_entropy, class_error
using ProtoGrad: repeatedly, is_logstep
using ProtoGrad: Adam, within_training_mode, init, eval_with_pullback, step!
using StatsBase: sample
using Serialization: serialize, deserialize
using Zygote

train_data = MNIST(Float32, :train)
train_x, train_y = train_data.features, train_data.targets
test_data = MNIST(Float32, :test)
test_x, test_y = test_data.features, test_data.targets

train_x_with_channel = reshape(train_x, size(train_x)[1:2]..., 1, size(train_x)[3])
train_y_onehot = onehotbatch(train_y, 0:9)

height, width, input_channels, dataset_size = size(train_x_with_channel)
output_channels = 16
num_classes = 10

out_conv_size = (div(height, 4) - 3, div(width, 4) - 3, 16)

model = Compose(
    Dropout(0.1),
    Conv(input_channels => 6, (5, 5)),
    x -> relu.(x),                       # can also use ReLU()
    x -> maxpool(x, (2, 2)),             # can also use MaxPool((2, 2))
    Dropout(0.1),
    Conv(6 => output_channels, (5, 5)),
    x -> relu.(x),
    x -> maxpool(x, (2, 2)),
    x -> reshape(x, :, size(x, 4)),      # can also use Reshape((...))
    Dropout(0.1),
    Linear(prod(out_conv_size) => 120),
    x -> relu.(x),
    Dropout(0.1),
    Linear(120 => 84),
    x -> relu.(x),
    Dropout(0.1),
    Linear(84 => num_classes),
    softmax,
)

batch_size = 128
num_batches = 10_000

training_batches = repeatedly(num_batches) do
    idx = sample(1:size(train_x_with_channel)[end], batch_size; replace = false)
    return (train_x_with_channel[:, :, :, idx], train_y_onehot[:, idx])
end

optimizer = Adam(; stepsize = 1e-3)
state = init(optimizer, model)

within_training_mode() do
    for (idx, (input, label)) in enumerate(training_batches)
        val, pb = eval_with_pullback(model, :Zygote) do m
            output = m(input)
            return cross_entropy(output, label)
        end
        if is_logstep(idx)
            @info "Training loss" step = idx loss = val
        end
        grad = pb()
        step!(state, grad)
    end
end

model_path = joinpath(@__DIR__, "serialized_conv_model")
println("saving model to $(model_path)")
serialize(model_path, model)

m_deserialized = deserialize(model_path)

test_x_with_channel = reshape(test_x, size(test_x)[1:2]..., 1, size(test_x)[3])
test_y_onehot = onehotbatch(test_y, 0:9)

err = class_error(m_deserialized(test_x_with_channel), test_y_onehot)

@info "Test error" error = err
