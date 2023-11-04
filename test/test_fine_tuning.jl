using ProtoGrad: GradientDescent, init, eval_with_pullback, mse, step!
using Test

@testset "Fine-tuning" begin
    @testset "$(T)" for T in [Float32, Float64]
        input_size, hidden_size, output_size = 3, 2, 1
        batch_size = 4

        m1 = ProtoGrad.Linear(T, input_size => hidden_size)
        m3 = ProtoGrad.Linear(T, hidden_size => output_size)

        function get_complete_model(b)
            return ProtoGrad.Compose(m1, ProtoGrad.ReLU(), b, x -> ProtoGrad.relu.(x), m3)
        end

        m2 = ProtoGrad.Linear(T, hidden_size => hidden_size)
        m2_orig = copy(m2)

        W_true = randn(T, output_size, input_size)
        b_true = randn(T, output_size)

        input = randn(T, input_size, batch_size)
        label = W_true * input .+ b_true + randn(T, output_size, batch_size)

        optimizer = GradientDescent(; stepsize = 0.1)
        state = init(optimizer, m2)

        for batch_idx = 1:10
            val, pb = eval_with_pullback(m2, :Zygote) do m
                full_m = get_complete_model(m)
                output = full_m(input)
                return mse(output, label)
            end
            grad = pb()
            step!(state, grad)
        end

        @test_skip m2.W != m2_orig.W
        @test_skip m2.b != m2_orig.b
    end
end
