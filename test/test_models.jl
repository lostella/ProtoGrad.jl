using ProtoGrad
using ProtoGrad: Model, Linear, Conv, ReLU, relu, softmax, Dropout, Compose, SupervisedObjective, mse
using LinearAlgebra
using Serialization
using Test

@testset "Basic operations" begin
    @testset "$(T)" for T in [Float16, Float32, Float64]
        @testset "$(name)" for (name, m) in [
            ("Linear", Linear(T, 3=>2)),
            ("Compose", Compose(Linear(T, 4=>3), ReLU(), Linear(T, 3=>2), x -> relu.(x)))
        ]
            vec_m = vec(m)

            @test eltype(vec_m) == T

            @test length(ProtoGrad.overlap(m, m)) == length(ProtoGrad.allparams(m))

            res = m + m
            @test typeof(res) == typeof(m)
            vec_res = vec(res)
            @test eltype(vec_res) == T
            @test vec_res == vec_m + vec_m

            res .= m .+ m
            @test typeof(res) == typeof(m)
            vec_res = vec(res)
            @test eltype(vec_res) == T
            @test vec_res == vec_m + vec_m

            res = m - m
            @test typeof(res) == typeof(m)
            vec_res = vec(res)
            @test eltype(vec_res) == T
            @test vec_res == vec_m - vec_m

            res .= m .- m
            @test typeof(res) == typeof(m)
            vec_res = vec(res)
            @test eltype(vec_res) == T
            @test vec_res == vec_m - vec_m

            res = 2 * m
            @test typeof(res) == typeof(m)
            vec_res = vec(res)
            @test eltype(vec_res) == T
            @test vec_res == 2 * vec_m

            res .= 2 .* m
            @test typeof(res) == typeof(m)
            vec_res = vec(res)
            @test eltype(vec_res) == T
            @test vec_res == 2 * vec_m

            res = m / 3
            @test typeof(res) == typeof(m)
            vec_res = vec(res)
            @test eltype(vec_res) == T
            @test vec_res == vec_m / 3

            res .= m ./ 3
            @test typeof(res) == typeof(m)
            vec_res = vec(res)
            @test eltype(vec_res) == T
            @test vec_res == vec_m / 3

            mp1 = m .+ 1
            dot_m_mp1 = dot(m, mp1)
            @test typeof(dot_m_mp1) == T
            @test dot_m_mp1 ≈ dot(vec_m, vec(mp1))

            mcopy = copy(m)
            @test typeof(mcopy) == typeof(m)
            vec_mcopy = vec(mcopy)
            @test eltype(vec_mcopy) == T
            @test vec_mcopy == vec_m
            @test length(ProtoGrad.overlap(m, mcopy)) == 0

            msimilar = similar(m)
            @test typeof(msimilar) == typeof(m)
            vec_msimilar = vec(msimilar)
            @test eltype(vec_msimilar) == T

            mzero = zero(m)
            @test typeof(mzero) == typeof(m)
            vec_mzero = vec(mzero)
            @test eltype(vec_mzero) == T
            @test all(vec_mzero .== T(0))
        end
    end
end

@testset "Gradient" begin
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

@testset "Serde" begin
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
