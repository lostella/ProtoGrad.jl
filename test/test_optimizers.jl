using Base.Iterators: repeated
using ProtoGrad: Linear, Compose, ReLU, SupervisedObjective, mse
using ProtoGrad: GradientDescent, Nesterov, BarzilaiBorwein, Adam
using LinearAlgebra
using Test

@testset "Optimizers" begin

    @testset "Linear ($(T))" for T in [Float32, Float64]

        input_size, output_size = 10, 1
        batch_size = 5000

        W_true = randn(T, output_size, input_size)
        b_true = randn(T, output_size)

        x = randn(T, input_size, batch_size)
        y = W_true * x .+ b_true + randn(T, output_size, batch_size) / 10

        data_iter = Iterators.repeated((x, y))
        f = SupervisedObjective(mse, data_iter)
        Lf = 2 * opnorm(x * x') / batch_size
        stepsize = 1 / Lf

        m = Linear(randn(T, output_size, input_size), randn(T, output_size))

        @testset "Accuracy $(name)" for (name, optimizer) in [
            ("GradientDescent", GradientDescent(stepsize=stepsize)),
            ("Nesterov", Nesterov(stepsize=stepsize)),
        ]
            m0 = copy(m)
            seq = optimizer(m0, f)

            @test Base.IteratorSize(typeof(seq)) == Base.IsInfinite()

            for (it, m_it) in enumerate(seq)
                if it >= 1000
                    @test typeof(m_it) == typeof(m)
                    @test isapprox(m_it.W, W_true, rtol=5e-2)
                    @test isapprox(m_it.b, b_true, rtol=5e-2)
                    break
                end
            end
        end

        @testset "Type stability $(name)" for (name, optimizer) in [
            ("GradientDescent(Float64)", GradientDescent(stepsize=1e-1)),
            ("GradientDescent(Float32)", GradientDescent(stepsize=1f-1)),
            ("Nesterov(Float64)", Nesterov(stepsize=1e-1)),
            ("Nesterov(Float32)", Nesterov(stepsize=1f-1)),
            ("BarzilaiBorwein(Float64)", BarzilaiBorwein(alpha=1e-1)),
            ("BarzilaiBorwein(Float32)", BarzilaiBorwein(alpha=1f-1)),
            ("Adam(Float64)", Adam(stepsize=1e-1)),
            ("Adam(Float32)", Adam(stepsize=1f-1)),
        ]
            m0 = copy(m)
            seq = optimizer(m0, f)

            for m_it in Iterators.take(seq, 5)
                @test typeof(m_it) == typeof(m)
            end
        end
    end
end
