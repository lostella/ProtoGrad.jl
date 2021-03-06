using MLDatasets: MNIST
using Flux: onehotbatch
using ProtoGrad: Linear, Dropout, Compose, relu, softmax
using ProtoGrad: SupervisedObjective, cross_entropy, class_error
using ProtoGrad: GradientDescent, forever, within_training_mode
using ProgressMeter: @showprogress
using StatsBase: sample
using Serialization: serialize, deserialize

train_x, train_y = MNIST.traindata(Float32, dir=joinpath(".", "datasets", "MNIST"))
test_x, test_y = MNIST.testdata(Float32, dir=joinpath(".", "datasets", "MNIST"))

train_x_flat = reshape(train_x, prod(size(train_x)[1:2]), size(train_x)[3])
train_y_onehot = onehotbatch(train_y, 0:9)

input_size, dataset_size = size(train_x_flat)
hidden_size = 100
num_classes = 10

m = Compose(
    Dropout(0.1),
    Linear(input_size=>hidden_size),
    x -> relu.(x),
    Dropout(0.1),
    Linear(hidden_size=>num_classes),
    softmax,
)

batch_size = 128

training_batches = forever() do
    idx = sample(1:size(train_x_flat)[end], batch_size, replace = false)
    return (train_x_flat[:, idx], train_y_onehot[:, idx])
end

f = SupervisedObjective(cross_entropy, training_batches)

optimizer = GradientDescent(stepsize=1e-1)
iterations = Iterators.Stateful(optimizer(m, f))

num_epochs = 100
num_batches_per_epoch = 500

within_training_mode() do
    @showprogress for epoch in 1:num_epochs
        for output in Iterators.take(iterations, num_batches_per_epoch)
            global m = output.solution
        end
    end
end

model_filename = "./serialized_mlp_model"
println("saving model to $(model_filename)")
serialize(model_filename, m)

m_deserialized = deserialize(model_filename)

test_x_flat = reshape(test_x, prod(size(test_x)[1:2]), size(test_x)[3])
test_y_onehot = onehotbatch(test_y, 0:9)

err = class_error(m_deserialized(test_x_flat), test_y_onehot)

println("test error = $(err * 100)%")
