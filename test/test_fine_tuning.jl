using ProtoGrad
using Test

@testset "Model fine-tuning" begin
    T = Float64

    input_size, hidden_size, output_size = 3, 2, 1
    batch_size = 4

    W1 = randn(T, hidden_size, input_size)
    b1 = randn(T, hidden_size)
    m1 = ProtoGrad.Linear(W1, b1)

    W2 = randn(T, hidden_size, hidden_size)
    b2 = randn(T, hidden_size)
    m2 = ProtoGrad.Linear(W2, b2)
    m2_orig = copy(m2)

    W3 = randn(T, output_size, hidden_size)
    b3 = randn(T, output_size)
    m3 = ProtoGrad.Linear(W3, b3)

    function get_complete_model(b)
        ProtoGrad.Compose(
            m1,
            ProtoGrad.ReLU(),
            b,
            x -> ProtoGrad.relu.(x),
            m3
        )
    end

    W_true = randn(T, output_size, input_size)
    b_true = randn(T, output_size)

    x = randn(T, input_size, batch_size)
    y = W_true * x .+ b_true + randn(T, output_size, batch_size)

    data_iter = Iterators.repeated((x, y))
    f = ProtoGrad.SupervisedObjective(ProtoGrad.mse, data_iter) ∘ get_complete_model

    optimizer = ProtoGrad.GradientDescent(stepsize=0.1)

    m2_final = nothing

    for output in Iterators.take(optimizer(m2, f), 10)
        m2_final = output.model
    end

    @test m2_final.W != m2_orig.W
    @test m2_final.b != m2_orig.b

end