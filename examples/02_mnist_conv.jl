using MLDatasets: MNIST
using Flux: onehotbatch
using ProtoGrad: Conv, Linear, Compose, maxpool, relu, softmax
using ProtoGrad: SupervisedObjective, forever, cross_entropy, class_error
using ProtoGrad: Adam
using ProgressMeter: @showprogress
using StatsBase: sample
using Serialization: serialize, deserialize

train_x, train_y = MNIST.traindata(Float32, dir=joinpath(".", "datasets", "MNIST"))
test_x, test_y = MNIST.testdata(Float32, dir=joinpath(".", "datasets", "MNIST"))

train_x_with_channel = reshape(train_x, size(train_x)[1:2]..., 1, size(train_x)[3])
train_y_onehot = onehotbatch(train_y, 0:9)

height, width, input_channels, dataset_size = size(train_x_with_channel)
output_channels = 16
num_classes = 10

out_conv_size = (div(height, 4) - 3, div(width, 4) - 3, 16)

m = Compose(
    Conv(input_channels=>6, (5, 5)),
    x -> relu.(x),                       # can also use ReLU()
    x -> maxpool(x, (2, 2)),             # can also use MaxPool((2, 2))
    Conv(6=>output_channels, (5, 5)),
    x -> relu.(x),
    x -> maxpool(x, (2, 2)),
    x -> reshape(x, :, size(x, 4)),      # can also use Reshape((...))
    Linear(prod(out_conv_size)=>120),
    x -> relu.(x),
    Linear(120=>84),
    x -> relu.(x),
    Linear(84=>num_classes),
    softmax,
)

batch_size = 128

training_batches = (
    begin
        idx = sample(1:size(train_x_with_channel)[end], batch_size, replace = false)
        (train_x_with_channel[:, :, :, idx], train_y_onehot[:, idx])
    end
    for _ in forever
)

f = SupervisedObjective(cross_entropy, training_batches)

optimizer = Adam()
iterations = Iterators.Stateful(optimizer(m, f))

num_epochs = 20
num_batches_per_epoch = 500

@showprogress for epoch in 1:num_epochs
    for m_it in Iterators.take(iterations, num_batches_per_epoch)
        global m = m_it
    end
end

model_filename = "./serialized_conv_model"
println("saving model to $(model_filename)")
serialize(model_filename, m)

m_deserialized = deserialize(model_filename)

test_x_with_channel = reshape(test_x, size(test_x)[1:2]..., 1, size(test_x)[3])
test_y_onehot = onehotbatch(test_y, 0:9)

err = class_error(m_deserialized(test_x_with_channel), test_y_onehot)

println("test error = $(err * 100)%")
