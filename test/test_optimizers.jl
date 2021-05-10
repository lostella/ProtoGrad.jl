using Base.Iterators: repeated, take, enumerate
using ProtoGrad
using ProtoGrad: Linear, Compose, ReLU, SupervisedObjective, mse
using LinearAlgebra
using Test

@testset "Quadratic objective" begin
    Q = Diagonal([1e1, 1e-1, 1e0, 1e0, 1e0, 1e0, 1e0, 1e0, 1e0, 1e0])
    q = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    objective = x -> 0.5*dot(x, Q*x) + dot(x, q)

    x_star = -Q\q
    f_star = objective(x_star)
    L = opnorm(Q)

    w0 = zeros(10)

    @testset "$(name)" for (name, optimizer) in [
        "GradientDescent" => ProtoGrad.GradientDescent(stepsize=1/L),
    ]
        for (k, output) in enumerate(take(optimizer(w0, objective), 200))
            @test objective(output.solution) - f_star <= 2 * L * norm(x_star)^2 / (2 * (k - 1))
        end
    end

    @testset "$(name)" for (name, optimizer) in [
        "Nesterov" => ProtoGrad.Nesterov(stepsize=1/L),
    ]
        for (k, output) in enumerate(take(optimizer(w0, objective), 200))
            @test objective(output.solution) - f_star <= 2 * L * norm(x_star)^2 / k^2
        end
    end

    @testset "$(name)" for (name, optimizer) in [
        "Polyak" => ProtoGrad.Polyak(stepsize=1/L, momentum=0.5),
        "RMSProp" => ProtoGrad.RMSProp(stepsize=1/L, alpha=0.99, epsilon=1e-8),
        "Adam" => ProtoGrad.Adam(stepsize=1/L, beta1=0.9, beta2=0.999, epsilon=1e-8),
    ]
        max_iter = 1000

        iterations = take(optimizer(w0, objective), max_iter)
        solution = ProtoGrad.last(iterations).solution

        @test objective(solution) <= f_star + 0.01 * abs(f_star)
    end
end

@testset "Supervised objective" begin

    @testset "Linear model ($(T))" for T in [Float32, Float64]

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
            "GradientDescent" => ProtoGrad.GradientDescent(stepsize=stepsize),
            "Polyak" => ProtoGrad.Polyak(stepsize=stepsize, momentum=0.5),
            "Nesterov" => ProtoGrad.Nesterov(stepsize=stepsize),
            "BarzilaiBorwein" => ProtoGrad.BarzilaiBorwein(alpha=stepsize),
            "AdaGrad" => ProtoGrad.AdaGrad(stepsize=stepsize),
            "RMSProp" => ProtoGrad.RMSProp(stepsize=stepsize),
            "AdaDelta" => ProtoGrad.AdaDelta(),
            "Adam" => ProtoGrad.Adam(stepsize=stepsize),
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
            "GradientDescent" => ProtoGrad.GradientDescent(stepsize=stepsize),
            "Polyak" => ProtoGrad.Polyak(stepsize=stepsize, momentum=0.5),
            "Nesterov" => ProtoGrad.Nesterov(stepsize=stepsize),
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
