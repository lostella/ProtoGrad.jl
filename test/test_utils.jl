using ProtoGrad
using Test

@testset "Itertools" begin
    @testset "$(typeof(itr))" for (itr, exp) in [
        ([1, 2, 3, 4, 5], 5),
        (1:10, 10),
    ]
        @test ProtoGrad.last(itr) == exp
    end
end

@testset "Training mode" begin
    @test ProtoGrad.is_training_mode() == false

    ProtoGrad.within_training_mode() do
        @test ProtoGrad.is_training_mode() == true
    end

    @test ProtoGrad.is_training_mode() == false
end
