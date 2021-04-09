using ProtoGrad
using Test

@testset "Fine-tuning" begin
    @testset "$(T)" for T in [Float32, Float64]
        input_size, hidden_size, output_size = 3, 2, 1
        batch_size = 4

        m1 = ProtoGrad.Linear(T, input_size=>hidden_size)
        m3 = ProtoGrad.Linear(T, hidden_size=>output_size)

        function get_complete_model(b)
            ProtoGrad.Compose(
                m1,
                ProtoGrad.ReLU(),
                b,
                x -> ProtoGrad.relu.(x),
                m3
            )
        end

        m2 = ProtoGrad.Linear(T, hidden_size=>hidden_size)
        m2_orig = copy(m2)

        W_true = randn(T, output_size, input_size)
        b_true = randn(T, output_size)

        x = randn(T, input_size, batch_size)
        y = W_true * x .+ b_true + randn(T, output_size, batch_size)

        data_iter = Iterators.repeated((x, y))
        f = ProtoGrad.SupervisedObjective(ProtoGrad.mse, data_iter) ∘ get_complete_model

        optimizer = ProtoGrad.GradientDescent(stepsize=0.1)

        m2_final = nothing

        for output in Iterators.take(optimizer(m2, f), 10)
            @test length(ProtoGrad.overlap(output.solution, m2)) == 0
            @test length(ProtoGrad.overlap(output.solution, m2_orig)) == 0
            m2_final = output.solution
        end

        @test length(ProtoGrad.overlap(m2_final, m2_orig)) == 0
        @test m2_final.W != m2_orig.W
        @test m2_final.b != m2_orig.b
    end
end
