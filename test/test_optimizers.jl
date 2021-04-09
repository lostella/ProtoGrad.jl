using Base.Iterators: repeated
using ProtoGrad: Linear, Compose, ReLU, SupervisedObjective, mse
using ProtoGrad: GradientDescent, Nesterov, BarzilaiBorwein, Adam
using LinearAlgebra
using Test

using Base.Iterators: take
using ProtoGrad
using LinearAlgebra
using Test

@testset "Quadratic objective" begin
    Q = Diagonal([1e1, 1e-1, 1e0, 1e0, 1e0, 1e0, 1e0, 1e0, 1e0, 1e0])
    q = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    objective = x -> 0.5*dot(x, Q*x) + dot(x, q)

    x_star = -Q\q
    f_star = objective(x_star)

    w0 = zeros(10)

    max_iter = 1000

    @testset "$(name)" for (name, optimizer) in [
        "GradientDescent" => ProtoGrad.GradientDescent(stepsize=1/opnorm(Q)),
        "Nesterov" => ProtoGrad.Nesterov(stepsize=1/opnorm(Q)),
        "Adam" => ProtoGrad.Adam(stepsize=1e-1, beta1=0.9, beta2=0.999, epsilon=1e-8),
    ]
        iterations = take(optimizer(w0, objective), max_iter)
        solution = ProtoGrad.last(iterations).solution

        @test objective(solution) <= f_star + 0.01 * abs(f_star)
    end
end

@testset "Supervised objective" begin

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

        @testset "Basic checks $(name)" for (name, optimizer) in [
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
            seq = optimizer(m, f)

            @test Base.IteratorSize(typeof(seq)) == Base.IsInfinite()

            for output in Iterators.take(seq, 5)
                @test length(ProtoGrad.overlap(m, output.solution)) == 0
                @test typeof(output.solution) == typeof(m)
            end

            @test vec(m) == vec(m0)
        end

        @testset "Accuracy $(name)" for (name, optimizer) in [
            ("GradientDescent", GradientDescent(stepsize=stepsize)),
            ("Nesterov", Nesterov(stepsize=stepsize)),
        ]
            seq = optimizer(m, f)

            for (it, output) in enumerate(seq)
                if it >= 1000
                    @test isapprox(output.solution.W, W_true, rtol=5e-2)
                    # @test isapprox(output.solution.b, b_true, rtol=1e-1)
                    break
                end
            end
        end
    end
end
