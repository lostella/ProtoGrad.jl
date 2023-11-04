using Base.Iterators: repeated, take, enumerate
using ProtoGrad: GradientDescent, NesterovMethod, HeavyBallMethod, Adam
using ProtoGrad: init, eval_with_pullback, step!
using ProtoGrad: Linear, Compose, ReLU, mse
using LinearAlgebra
using Test

@testset "Quadratic objective" begin
    Q = Diagonal([1e1, 1e-1, 1e0, 1e0, 1e0, 1e0, 1e0, 1e0, 1e0, 1e0])
    q = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    objective = x -> 0.5 * dot(x, Q * x) + dot(x, q)

    x_star = -Q \ q
    f_star, pb = eval_with_pullback(objective, x_star, :Zygote)
    grad_star = pb()
    L = opnorm(Q)

    @test isapprox(norm(grad_star), 0)

    @testset "$(name)" for (name, optimizer) in
                           ["GradientDescent" => GradientDescent(; stepsize = 1 / L)]
        x = zeros(10)
        state = init(optimizer, x)
        for k = 1:200
            val, pb = eval_with_pullback(objective, x, :Zygote)
            grad = pb()
            @test val - f_star <= 2 * L * norm(x_star)^2 / (2 * (k - 1))
            step!(state, grad)
        end
    end

    @testset "$(name)" for (name, optimizer) in
                           ["Nesterov" => NesterovMethod(stepsize = 1 / L)]
        x = zeros(10)
        state = init(optimizer, x)
        for k = 1:200
            val, pb = eval_with_pullback(objective, x, :Zygote)
            grad = pb()
            @test val - f_star <= 2 * L * norm(x_star)^2 / k^2
            step!(state, grad)
        end
    end

    @testset "$(name)" for (name, optimizer) in [
        "Polyak" => HeavyBallMethod(stepsize = 1 / L, momentum = 0.5),
        "Adam" => Adam(stepsize = 1 / L, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8),
    ]
        x = zeros(10)
        state = init(optimizer, x)
        for k = 1:1000
            val, pb = eval_with_pullback(objective, x, :Zygote)
            grad = pb()
            step!(state, grad)
        end
        @test objective(x) <= f_star + 0.01 * abs(f_star)
    end
end

@testset "Linear model ($(T))" for T in [Float32, Float64]
    input_size, output_size = 10, 1
    batch_size = 5000

    W_true = randn(T, output_size, input_size)
    b_true = randn(T, output_size)

    input = randn(T, input_size, batch_size)
    label = W_true * input .+ b_true + randn(T, output_size, batch_size) / 10

    Lf = 2 * opnorm(input * input') / batch_size
    stepsize = 1 / Lf

    @testset "Accuracy $(name)" for (name, optimizer) in [
        "GradientDescent" => GradientDescent(stepsize = stepsize),
        "HeavyBall" => HeavyBallMethod(stepsize = stepsize, momentum = T(0.5)),
        "Nesterov" => NesterovMethod(stepsize = stepsize),
    ]
        model = Linear(randn(T, output_size, input_size), randn(T, output_size))
        state = init(optimizer, model)
        for it = 1:1000
            val, pb = eval_with_pullback(model, :Zygote) do m
                output = m(input)
                return mse(output, label)
            end
            if it >= 1000
                @test isapprox(model.W, W_true, rtol = 5e-2)
                break
            end
            grad = pb()
            step!(state, grad)
        end
    end
end
