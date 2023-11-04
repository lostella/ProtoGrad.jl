using Literate
using ProtoGrad

Literate.markdown(
    joinpath(@__DIR__, "src", "README.jl"),
    joinpath(dirname(pathof(ProtoGrad)), "..");
    execute = true,
    flavor = Literate.CommonMarkFlavor(),
)
