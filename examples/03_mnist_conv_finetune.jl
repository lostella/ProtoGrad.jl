using MLDatasets: MNIST
using Flux: onehotbatch
using ProtoGrad: Conv, Linear, Dropout, Compose, maxpool, relu, softmax
using ProtoGrad: SupervisedObjective, cross_entropy, class_error
using ProtoGrad: Adam, forever, within_training_mode
using ProgressMeter: @showprogress
using StatsBase: sample
using Serialization: serialize, deserialize

train_x, train_y = MNIST.traindata(Float32, dir=joinpath(".", "datasets", "MNIST"))
test_x, test_y = MNIST.testdata(Float32, dir=joinpath(".", "datasets", "MNIST"))

train_x_with_channel = reshape(train_x, size(train_x)[1:2]..., 1, size(train_x)[3])
train_y_onehot = onehotbatch(train_y, 0:9)

input_model_filename = "./serialized_conv_model"
println("loading model from $(input_model_filename)")
m_original = deserialize(input_model_filename)

function get_complete_model(m_top)
    return Compose(
        m_original.layers[1:end-3]...,
        m_top.layers...
    )
end

m_top = Compose(m_original.layers[end-2:end]...)

batch_size = 128

training_batches = forever() do
    idx = sample(1:size(train_x_with_channel)[end], batch_size, replace = false)
    return (train_x_with_channel[:, :, :, idx], train_y_onehot[:, idx])
end

f = SupervisedObjective(cross_entropy, training_batches) ∘ get_complete_model

optimizer = Adam()
iterations = Iterators.Stateful(optimizer(m_top, f))

num_epochs = 5
num_batches_per_epoch = 500

within_training_mode() do
    @showprogress for epoch in 1:num_epochs
        for output in Iterators.take(iterations, num_batches_per_epoch)
            global m_top = output.model
        end
    end
end

m = get_complete_model(m_top)

model_filename = "./serialized_conv_finetuned_model"
println("saving model to $(model_filename)")
serialize(model_filename, m)

m_deserialized = deserialize(model_filename)

test_x_with_channel = reshape(test_x, size(test_x)[1:2]..., 1, size(test_x)[3])
test_y_onehot = onehotbatch(test_y, 0:9)

err = class_error(m_deserialized(test_x_with_channel), test_y_onehot)

println("test error = $(err * 100)%")
