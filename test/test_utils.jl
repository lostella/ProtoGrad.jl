using ProtoGrad
using Test

@testset "Itertools" begin
    @testset "$(typeof(itr))" for (itr, exp) in [([1, 2, 3, 4, 5], 5), (1:10, 10)]
        @test ProtoGrad.last(itr) == exp
    end

    @testset "Settable" begin
        v = 0.0
        b = ProtoGrad.Settable(v)
        for value in Iterators.take(b, 10)
            @test value == v
            v = rand()
            ProtoGrad.set!(b, v)
        end
    end
end

@testset "Training mode" begin
    @test ProtoGrad.is_training_mode() == false

    ProtoGrad.within_training_mode() do
        @test ProtoGrad.is_training_mode() == true
    end

    @test ProtoGrad.is_training_mode() == false
end
