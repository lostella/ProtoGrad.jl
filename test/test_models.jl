using ProtoGrad
using ProtoGrad: Model, Linear, Conv, ReLU, relu, softmax, Dropout, Compose, SupervisedObjective, mse
using LinearAlgebra
using Serialization
using Test

@testset "Model operations" begin
    @testset "Broadcast ($(T))" for T in [Float32, Float64]
        struct ModelA <: Model
            attr1
            attr2
        end

        n = 3

        objA = ModelA(ones(T, n), ones(T, n))

        @test vec(objA) == ones(T, 2*n)

        destA = objA + objA
        @test typeof(destA) == typeof(objA)
        @test vec(destA) == vec(objA) + vec(objA)

        destA .= objA .+ objA
        @test typeof(destA) == typeof(objA)
        @test vec(destA) == vec(objA) + vec(objA)

        destA = 2 * objA
        @test typeof(destA) == typeof(objA)
        @test vec(destA) == 2 * vec(objA)

        destA .= 2 .* objA
        @test typeof(destA) == typeof(objA)
        @test vec(destA) == 2 * vec(objA)

        struct ModelB{M <: Model} <: Model
            model::M
        end

        objB = ModelB(ModelA(ones(T, n), ones(T, n)))

        @test vec(objB) == ones(T, 2*n)

        destB = objB + objB
        @test typeof(destB) == typeof(objB)
        @test vec(destB) == vec(objB) + vec(objB)

        destB .= objB .+ objB
        @test typeof(destB) == typeof(objB)
        @test vec(destB) == vec(objB) + vec(objB)

        destB = 2 * objB
        @test typeof(destB) == typeof(objB)
        @test vec(destB) == 2 * vec(objB)

        destB .= 2 .* objB
        @test typeof(destB) == typeof(objB)
        @test vec(destB) == 2 * vec(objB)

        # TODO broadcast doesn't work with the following
        # TODO need to figure out why

        # struct ModelC <: Model
        #     models::Tuple
        # end

        # objC = ModelC((
        #     ModelA(ones(n), ones(n)),
        #     ModelB(ModelA(ones(n), ones(n))),
        # ))

        # @test vec(objC) == ones(4*n)

        # destC = objC .+ objC
        # @test typeof(destC) == typeof(objC)
        # @test vec(destC) == vec(objC) + vec(objC)

        # destC .= objC .+ objC
        # @test typeof(destC) == typeof(objC)
        # @test vec(destC) == vec(objC) + vec(objC)

        # destC = 2 .* objC
        # @test typeof(destC) == typeof(objC)
        # @test vec(destC) == 2 * vec(objC)

        # destC .= 2 .* objC
        # @test typeof(destC) == typeof(objC)
        # @test vec(destC) == 2 * vec(objC)    
    end

    @testset "Linear ($(T))" for T in [Float32, Float64]
        W1 = randn(T, 1, 10)
        b1 = randn(T, 1)
        m1 = Linear(W1, b1)

        @test typeof(vec(m1)) == Vector{T}

        W2 = randn(T, 1, 10)
        b2 = randn(T, 1)
        m2 = Linear(W2, b2)

        m_sum = m1 + m2
        @test typeof(m_sum) == typeof(m1)
        @test vec(m_sum) == vec(m1) + vec(m2)

        m_diff = m1 .- m2
        @test typeof(m_diff) == typeof(m1)
        @test vec(m_diff) == vec(m1) - vec(m2)

        m_scaled1 = 3 * m1
        @test typeof(m_scaled1) == typeof(m1)
        @test vec(m_scaled1) == 3 * vec(m1)

        m_scaled2 = m1 * 4
        @test typeof(m_scaled2) == typeof(m1)
        @test vec(m_scaled2) == vec(m1) * 4

        m_scaled3 = m1 / 5
        @test typeof(m_scaled3) == typeof(m1)
        @test vec(m_scaled3) == vec(m1) / 5
    end

    @testset "Compose ($(T))" for T in [Float32, Float64]
        W1 = randn(T, 4, 10)
        b1 = randn(T, 4)
        m1 = Linear(W1, b1)

        W2 = randn(T, 2, 4)
        b2 = randn(T, 2)
        m2 = Linear(W2, b2)

        m = Compose(m1, ReLU(), m2, x -> relu.(x))

        @test typeof(vec(m)) == Vector{T}

        x = randn(T, 10, 16)
        y = m(x)

        @test size(y) == (2, 16)
        @test all(y .>= 0)

        m_sum = m + m
        @test typeof(m_sum) == typeof(m)
        @test vec(m_sum) == vec(m) + vec(m)
        @test size(m_sum(x)) == (2, 16)

        m_diff = m - m
        @test typeof(m_diff) == typeof(m)
        @test vec(m_diff) == vec(m) - vec(m)
        @test size(m_diff(x)) == (2, 16)

        m_scaled1 = 3 * m
        @test typeof(m_scaled1) == typeof(m)
        @test vec(m_scaled1) == 3 * vec(m)
        @test size(m_scaled1(x)) == (2, 16)

        m_scaled2 = m * 4
        @test typeof(m_scaled2) == typeof(m)
        @test vec(m_scaled2) == vec(m) * 4
        @test size(m_scaled2(x)) == (2, 16)

        m_scaled3 = m / 5
        @test typeof(m_scaled3) == typeof(m)
        @test vec(m_scaled3) == vec(m) / 5
        @test size(m_scaled3(x)) == (2, 16)
    end

end

@testset "Model gradient" begin
    @testset "Linear ($(T))" for T in [Float32, Float64]
        input_size, output_size = 1000, 1
        batch_size = 50

        W = randn(T, output_size, input_size)
        b = randn(T, output_size)
        m = Linear(W, b)

        x = randn(T, input_size, batch_size)
        y = W * x .+ b + randn(T, output_size, batch_size)

        data_iter = Iterators.repeated((x, y))
        f = SupervisedObjective(mse, data_iter)

        grad, out = ProtoGrad.gradient(f, m)

        @test typeof(out) <: Number  # TODO improve this assertion
        @test typeof(grad) == typeof(m)

        @test out ≈ (norm(m.W * x .+ m.b - y) ^ 2) / batch_size
        @test grad.W ≈ (2 / batch_size) * ((m.W * x .+ m.b - y) * x')
        @test grad.b ≈ (2 / batch_size) * (m.W * x .+ m.b - y) * ones(batch_size)

        m .= m .- 3.0 .* grad
    end

    @testset "Compose ($(T))" for T in [Float32, Float64]
        input_size, hidden_size, output_size = 1000, 50, 1
        batch_size = 64

        W1 = randn(T, hidden_size, input_size)
        b1 = randn(T, hidden_size)
        m1 = Linear(W1, b1)

        W2 = randn(T, output_size, hidden_size)
        b2 = randn(T, output_size)
        m2 = Linear(W2, b2)

        m = Compose(m1, ReLU(), m2, x -> relu.(x))

        W_true = randn(T, output_size, input_size)
        b_true = randn(T, output_size)

        x = randn(T, input_size, batch_size)
        y = W_true * x .+ b_true + randn(T, output_size, batch_size)

        data_iter = Iterators.repeated((x, y))
        f = SupervisedObjective(mse, data_iter)

        grad, out = ProtoGrad.gradient(f, m)

        @test typeof(out) <: Number  # TODO improve this assertion
        @test typeof(grad) == typeof(m)

        m .= m .- 3.0 .* grad
    end

    @testset "Custom ($(T))" for T in [Float32, Float64]
        struct CustomModel <: Model
            m1
            act1
            m2
            act2
        end

        (m::CustomModel)(x) = (m.act2 ∘ m.m2 ∘ m.act1 ∘ m.m1)(x)

        input_size, hidden_size, output_size = 1000, 50, 1
        batch_size = 64

        W1 = randn(T, hidden_size, input_size)
        b1 = randn(T, hidden_size)
        m1 = Linear(W1, b1)

        W2 = randn(T, output_size, hidden_size)
        b2 = randn(T, output_size)
        m2 = Linear(W2, b2)

        m = CustomModel(m1, ReLU(), m2, x -> relu.(x))

        W_true = randn(T, output_size, input_size)
        b_true = randn(T, output_size)

        x = randn(T, input_size, batch_size)
        y = W_true * x .+ b_true + randn(T, output_size, batch_size)

        data_iter = Iterators.repeated((x, y))
        f = SupervisedObjective(mse, data_iter)

        grad, out = ProtoGrad.gradient(f, m)

        @test typeof(out) <: Number  # TODO improve this assertion
        @test typeof(grad) == typeof(m)

        m .= m .- 3.0 .* grad
    end

    @testset "Recurrent + Linear ($(T))" for T in [Float32, Float64]
        m = Compose(
            ProtoGrad.RNN(T, 3=>4, relu),
            Linear(4=>1)
        )

        x = randn(3, 20, 8)
        y = randn(1, 20, 8)

        data_iter = Iterators.repeated((x, y))
        f = SupervisedObjective(mse, data_iter)

        grad, out = ProtoGrad.gradient(f, m)

        @test typeof(out) <: Number  # TODO improve this assertion
        @test typeof(grad) == typeof(m)

        m .= m .- 3.0 .* grad
    end

end

@testset "Dropout" begin
    for T in [Float32, Float64]
        input_size, hidden_size, output_size = 1000, 50, 1
        batch_size = 64

        m = ProtoGrad.Dropout(0.1)

        x = randn(T, input_size, batch_size)
        y = randn(T, output_size, batch_size)

        data_iter = Iterators.repeated((x, y))
        f = model -> ProtoGrad.mse(model(x), y)

        grad, out = ProtoGrad.within_training_mode() do
            ProtoGrad.gradient(f, m)
        end

        @test typeof(out) <: Number  # TODO improve this assertion
        @test typeof(grad) == typeof(m)

        m .= m .- 3.0 .* grad
    end
end

@testset "Model serde" begin
    for (m, x) in [
        (Linear(Float32, 20=>10), randn(Float32, 20, 4)),
        (Linear(Float64, 20=>10), randn(Float64, 20, 4)),
        (Conv(Float32, 3=>5, (3, 3)), randn(Float32, 10, 10, 3, 4)),
        (Conv(Float64, 3=>5, (3, 3)), randn(Float64, 10, 10, 3, 4)),
        (
            Compose(
                Linear(20=>10),
                ReLU(),
                Linear(10=>5),
                x -> relu.(x),
                softmax
            ),
            randn(Float32, 20, 4)
        )
    ]
        out = m(x)

        (path, io) = mktemp()
        Serialization.serialize(io, m)
        close(io)

        io = open(path, "r")
        m_copy = Serialization.deserialize(io)
        close(io)

        # NOTE this is not guaranteed to work
        # e.g. when local definitions (such as functions) are serialized
        # @test typeof(m_copy) == typeof(m)

        out_copy = m_copy(x)

        @test out_copy == out
    end
end
