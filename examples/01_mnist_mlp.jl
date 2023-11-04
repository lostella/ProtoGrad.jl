using MLDatasets: MNIST
using Flux: onehotbatch
using ProtoGrad: Linear, Dropout, Compose, relu, softmax
using ProtoGrad: cross_entropy, class_error
using ProtoGrad: repeatedly, is_logstep
using ProtoGrad: GradientDescent, within_training_mode, init, eval_with_pullback, step!
using StatsBase: sample
using Serialization: serialize, deserialize
using Zygote

train_data = MNIST(Float32, :train)
train_x, train_y = train_data.features, train_data.targets
test_data = MNIST(Float32, :test)
test_x, test_y = test_data.features, test_data.targets

train_x_flat = reshape(train_x, prod(size(train_x)[1:2]), size(train_x)[3])
train_y_onehot = onehotbatch(train_y, 0:9)

input_size, dataset_size = size(train_x_flat)
hidden_size = 100
num_classes = 10

model = Compose(
    Dropout(0.1),
    Linear(input_size => hidden_size),
    x -> relu.(x),
    Dropout(0.1),
    Linear(hidden_size => num_classes),
    softmax,
)

batch_size = 128
num_batches = 50_000

training_batches = repeatedly(num_batches) do
    idx = sample(1:size(train_x_flat)[end], batch_size; replace = false)
    return (train_x_flat[:, idx], train_y_onehot[:, idx])
end

optimizer = GradientDescent(; stepsize = 1e-1)
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

model_path = joinpath(@__DIR__, "serialized_mlp_model")
println("saving model to $(model_path)")
serialize(model_path, model)

m_deserialized = deserialize(model_path)

test_x_flat = reshape(test_x, prod(size(test_x)[1:2]), size(test_x)[3])
test_y_onehot = onehotbatch(test_y, 0:9)

err = class_error(m_deserialized(test_x_flat), test_y_onehot)

@info "Test error" error = err
