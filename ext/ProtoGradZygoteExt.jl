module ProtoGradZygoteExt

using ProtoGrad
using Zygote: pullback

function ProtoGrad.eval_with_pullback(f, m, ::Val{:Zygote})
    out, pb = pullback(f, m)
    function reconstruct_zygote_pullback()
        raw_grad = pb(one(out))[1]
        return ProtoGrad.reconstruct(raw_grad, m)
    end
    return out, reconstruct_zygote_pullback
end

end
