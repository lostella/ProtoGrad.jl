using Literate
using ProtoGrad

Literate.markdown(joinpath(@__DIR__, "src", "README.jl"), joinpath(ProtoGrad |> pathof |> dirname, ".."), execute=true)
