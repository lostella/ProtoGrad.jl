# ProtoGrad.jl

[![Build status](https://github.com/lostella/ProtoGrad.jl/workflows/Test/badge.svg)](https://github.com/lostella/ProtoGrad.jl/actions?query=workflow%3ATest)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

ProtoGrad is an experimental Julia package to work with gradient-based model optimization, including (of course!) deep learning.
It aims at being simple, composable, and flexible.
This said, it's very much of a work-in-progress playground for ideas, so don't expect feature completeness or stability just yet.

The package builds on top of much more mature and popular libraries, above all [Zygote](https://github.com/FluxML/Zygote.jl) (for automatic differentiation) and [NNLib](https://github.com/FluxML/NNlib.jl) (providing common operators in deep learning).

Check out the [examples folder](./examples/) on how to use ProtoGrad to construct and train models, or keep following the present README to get a feeling for the package philosophy.

It all begins, naturally, with

````julia
using ProtoGrad
````

## Models

Models are just callable objects, whose type extends the `ProtoGrad.Model` abstract type.
The following (overly) simple example defines some type of linear model (a better version of this is `ProtoGrad.Linear`):

````julia
struct LinearModel <: ProtoGrad.Model
    A::Any
    b::Any
end

(m::LinearModel)(x) = m.A * x .+ m.b
````

All attributes of a model are interpreted as parameters to be optimized, and so gradients will be taken with respect to them. It is therefore assumed that all attributes are either
1. Numerical arrays,
2. Functions,
3. Other `Model` objects,
4. `Tuple`s of objects of the above types.

> **Note:** This means, for example, that hyper-paramenters cannot be stored as attributes.
> Some hyperparameters are implicit in the model structure (e.g. number of layers or units);
> otherwise, they can be stored as type parameters (as ["value types"](https://docs.julialang.org/en/v1/manual/types/#%22Value-types%22)).

Models defined this way get the structure of a vector space, for free:

````julia
m = LinearModel(randn(3, 5), randn(3))
m_scaled = 2 * m # this is also of type LinearModel
m_sum = m + m_scaled # this too
````

````
Main.var"##292".LinearModel
````

The dot-syntax for in-place assignment and loop fusion can also be used:

````julia
m_scaled .= 2 .* m
m_sum .= m .+ m_scaled
````

````
Main.var"##292".LinearModel
````

And you can take dot products too!

````julia
using LinearAlgebra
dot(m, m_sum)
````

````
27.56698106108678
````

## Objective functions

Training a model usually amounts to optimizing some objective function.
In principle, any custom function of the model will do.
For example, we can use the mean squared error

````julia
mean_squared_error(yhat, y) = sum((yhat .- y) .^ 2) / size(y)[end]
````

````
mean_squared_error (generic function with 1 method)
````

together with some data (here artificially generated, according to a random, noisy linear model)

````julia
A_original = randn(3, 5)
b_original = randn(3)
x = randn(5, 300)
y = A_original * x .+ b_original .+ randn(3, 300)
````

````
3×300 Matrix{Float64}:
 -0.0127804   0.87501   -3.71636    0.880898  -0.22888    3.48398   1.29364   0.115065  -3.05687   1.12702    2.49889  -0.750049   0.201259   2.66435     3.62157   -0.37745   -4.43161  -1.54655   -0.676121   4.73542  -0.530054  -1.55773   -1.25418  -2.66997   2.18242  -0.247722  -0.368092   1.35801     3.64573  -0.524075  -2.05085   4.06166  -0.349323   0.370862   2.7113   -1.12613  -0.272618  -1.9573    -1.94311    3.12749   -2.40577    1.80094  -1.33273   2.43399   -1.78341  -1.70952  -2.59597   0.219181   1.3303    0.329741   0.438782   1.79774   -3.59878   -1.58054   -0.142307   2.10504  -0.567967  -3.55972    -3.06548    0.718006   3.13294   -0.922412   2.6019    0.927639  2.63337   -3.59824    3.99913   0.144014  -2.1733   1.94323   -1.61539  -1.01775   3.46251   1.68835  -0.583456  -1.45855   3.04903   -1.71319   -1.15443   -0.456181   0.402848   0.312761  1.70906   -4.41435   0.29185   -0.642912  -3.3532    3.74804   -0.527566    4.87592    -0.026144  -0.621949  -0.109881  -4.33657    0.790791  -1.96567    1.93405    0.346101  -3.22151   0.367314   0.0733407   0.884018  -0.22212   2.14215   -4.12249   2.19278   2.64321    2.39857   2.6594    -3.28142    0.3646   -0.465708  0.986858   0.555068  -2.80781  -1.70147  6.95208  -2.33141   0.349133   0.901867   1.1712    -0.676677  -0.314246  -2.31113   0.350148   0.0991809  -0.683335   -2.36344   -0.737621   3.62776  1.85308   -0.181613  -1.39142   -0.1251   2.77461   -2.52641   -1.50651    2.98617  -0.183764  -2.17419   2.74174   0.649361  -2.2728   -0.312313   2.96936   2.14549  -1.22686   2.2331   -1.11995   -2.79686   -1.22388  -3.78298  -2.28157     -2.432      0.477566  1.39663   -0.415349   1.37119    1.49992  -3.01297    0.383269  -2.97362  -3.79563  -1.64702    0.270169  -1.80254  -1.05342   2.49049   0.257852   1.25912   -2.41714  -0.261056  -3.32531   0.770814   3.47718   3.53207    1.91369  -1.99618    2.02207   4.28609   -1.22012  -1.32978  -1.80895   0.222604  -2.2389    0.283643   0.734692   5.94879   2.6593     2.68186    1.81671   2.44069   -0.0974266   1.72138  -3.98022   0.715    -0.00353688  -1.07691    1.81639   1.6517    4.3451    0.251144   4.33738   4.01019  -0.038388   0.639271  -0.982071   2.80261  -1.03358  0.731103  -0.474898  -3.42722   2.91623  -2.20969   -0.790885  -1.54079   2.20005  -1.96096   1.26452   0.203823  -2.04519  -0.468539  -2.34095  -2.81808    4.10277  -0.863523   1.53198  -0.0965248  -0.635501   0.66941   2.97038    2.38469   2.76237  -4.52934   2.41332   -1.34411   -5.08311   0.550303  -1.55582   2.72161  -4.14647  -1.29288    0.189616   0.300006  -4.8365   -1.94738   1.20673  -1.04479    0.685914  -0.985226  0.251558  -1.7142     4.26661  -0.770436   1.80673   0.502264   2.21721  -0.904591  2.15408  -2.64462  2.28511    0.813357  -2.69705    0.119931  -1.21737   1.34665   1.70322  -2.64473   -0.744921   1.93647   -3.99576    0.509565  2.51512    1.50033   1.00954  -3.87959    1.37925    -2.69166    -0.0567109  -3.06404    -2.42034  -1.1468   -4.41309    2.80483   1.99937   0.239848   1.97077   0.883873  -0.819619  -1.56851    -2.15334   -0.318483   0.205195  -3.61331  -1.90389  -1.55819   -1.02501  -0.544636   0.0335983  -0.45265
 -1.75155    -4.09854   -8.45081   -4.35235    3.16879   -1.09147  -1.05099  -0.97481   -4.60412  -1.17868   -3.51637   0.274543  -2.13546    0.509944    0.243204   0.369729  -3.06434  -0.987334  -0.651721  -1.15117  -1.60262   -0.002578  -3.12706  -5.61117  -3.27747  -3.77768   -3.35824    0.0450191  -2.80837  -2.99294   -2.22177  -3.37684  -3.75315   -3.85296   -1.70209  -2.23225  -1.6301     1.47077   -2.39891   -5.06699   -5.88863   -5.61693  -7.71642  -0.889695  -3.27906  -3.21904  -3.57883  -5.07637   -3.68082  -5.3971    -3.81933   -5.26021   -3.3749    -4.99881   -2.36904   -3.99303  -1.30557    0.0888809  -0.836996   1.11392   -2.40532    2.52184   -1.46135  -0.763668  1.38072   -5.77965   -7.72453   0.315051  -5.50602  1.43985   -4.21471  -3.26705  -3.01378  -4.68627  -2.80903   -2.31601  -0.514964  -3.42146    2.07954    1.72439    1.41619   -5.09274   1.51432   -3.34304  -5.56787   -0.539649  -5.25413  -1.30595    0.46813    -0.0168354  -6.86988   -3.17497   -3.78461   -1.71893   -5.22363   -3.28168   -6.09002   -3.0346     1.30446  -2.74904   -2.47496    -0.438914   1.8651   -1.4025    -3.93555  -3.37295  -5.50054   -5.46636  -4.52577   -1.92246   -1.8494   -5.13779   0.275152   0.436219  -2.57483  -4.01671  4.14159   0.59712  -0.411385  -1.23873   -1.72234   -5.74654    1.73921   -3.17305  -3.81803    0.861162   -4.77063    -0.832944  -5.38865   -1.58031  2.03557   -3.19734   -0.193169  -3.30889  0.837127   0.833884  -2.72187   -3.09041  -2.05153    3.657    -1.93502   0.178592  -6.54055  -4.24832   -1.58878  -1.79714  -1.33482  -4.03666  -4.43943   -4.61413   -2.27226  -4.51673  -1.16362      0.244055  -2.67821   0.673829  -4.64092   -3.27862   -2.56354  -7.97674   -6.66319   -3.93653  -1.38644   0.380524  -5.03502   -5.2718   -4.61069  -6.3117   -5.01939   -1.94193   -2.78524  -3.05881   -3.19152  -2.65477   -3.22299  -1.63551   -3.04115  -1.28839   -0.62595   0.322003   2.99481  -7.14102  -5.08647  -0.85981   -5.3612    0.83419   -0.899557  -1.92049  -4.15565   -2.88527   -2.18362  -1.313      3.39404    -4.73054  -3.14755  -1.6579   -6.44849     -5.48353   -3.97057  -2.77703  -2.35477  -2.07363    1.75349  -3.10031  -6.07041   -5.39643   -9.92075   -4.59538  -1.62984  0.692023  -3.52106   -1.04905  -3.07536  -7.24328    1.53163   -5.2178   -3.66997  -3.4288   -5.97737  -1.59053   -4.3106    2.80258   -3.88494   0.968303  -3.62483  -3.63292   -1.9637   -0.366792   -1.71179   -5.01719  -1.58682    2.10614   1.1728   -3.47907   1.67634    0.096756  -3.06262  -0.521412  -8.68292  -5.91953  -1.60424  -0.296123  -1.85015   -5.09239   -4.47361  -2.34843  -3.80053  -3.29211   -2.19604   -1.87457   1.54526   -4.42549   -1.17173  -2.62552   -3.2781   -6.21904   -3.53637  -1.86489   1.38086  -6.34146  0.661481   0.488116  -1.41559   -0.792732   4.59659  -4.96944  -4.65995  -3.00781   -1.21892   -1.6273    -5.98664    0.674404  0.970331  -1.71651   1.41695  -4.20622   -2.19014    -3.74431    -1.12465    -3.37965    -2.45365   2.991     0.577795  -3.50005  -3.72935  -1.52785   -1.0183   -5.7044    -3.20982   -1.477      -2.07614   -2.62408   -1.46479   -6.22128  -4.43397   0.593822   2.13659  -3.89604   -4.08849    -0.875612
  0.544636    0.352614   0.947703   3.16107    0.431965   2.34735   4.23469  -3.45758    2.32268   0.339819   0.2319   -0.455574   0.342672  -0.0894179  -0.240974   0.21374    1.73803   1.06558   -0.468115   1.3644    1.22206   -2.32379    1.11084  -1.74199   3.57507   2.06176    0.613532  -0.301007    1.70003  -0.415912  -0.1052   -1.97267   2.39498   -1.82246   -2.52935   1.22288   0.167527   0.485382  -0.167962   0.948745  -0.135469  -2.69285   2.78811   1.12809    2.55033   3.25793  -3.20273  -1.09549    2.74782   0.490701  -2.18757    0.914631  -0.567978  -0.446793   3.36456    3.27623  -0.152601   0.829666    2.28763   -1.21242    0.470196  -3.4978     4.03642   0.699777  0.780959  -0.975619   3.44697  -1.96098   -2.42462  0.324809   1.07015   1.71728   2.6603    2.03935   0.747084  -1.08715  -0.543938  -0.403495  -0.102962  -0.11095   -0.840845  -0.621742  0.453373   3.11884  -0.236127   1.3833     1.53199   0.953667   0.0554594   3.93105     2.7704     1.49845   -1.13678   -0.693149   2.18139    0.169975   0.559242  -2.58319   -2.69989   1.16261    2.45208    -1.34625   -1.97861   0.803265   1.31874   3.87583  -0.392541  -1.13373   0.479034  -0.846679  -1.35078   4.42819   0.847526  -0.260605   1.11745   2.71234  1.19989   1.61025  -2.10108   -5.1439    -0.254541  -2.39733   -0.255461   2.05345  -1.99418   -0.172726    0.0234827   1.22752   -0.321964   2.95251  0.328228   0.747972   1.53926    2.91978  1.06371   -2.07999    0.826805   1.85865   0.530423  -4.19732  -1.02635  -3.44538   -1.76118   2.81535    4.49542  -1.3837    1.07369   3.41389  -0.426342  -0.944657   2.00314  -0.25493  -0.00944803  -2.65624   -1.97469   1.28938   -0.267113   0.465471   1.26622   0.657988   2.30168    1.66588   1.92814  -1.23453    1.46931    2.44841  -1.25907   3.04436   2.91099    0.261829  -2.13439   0.127197   3.35098   3.75285    2.71683   0.225209  -3.01345   0.398232  -0.604123  1.82749   -0.66078  -3.06627   2.09015  -4.2562    -0.274738  2.41339    3.45187    0.38946   0.129265   0.767873   4.5396    0.086942   0.549179    4.88025   1.36201   0.24464   1.19925     -0.394464   2.31138   3.02137   1.77033  -0.278727  -1.52173   4.2877    3.78644   -1.13039    3.84309   -0.40405   2.57206  1.22312    6.2875    -1.51017   3.38131  -0.890651   0.808098   3.38939   3.66101  -1.17162   2.67851  -0.394379   1.51557  -2.98429   -2.44933  -5.40169    2.95603  -0.513862   1.42918   1.31506     0.386542  -1.75127   0.102994  -1.40098  -1.66295   2.41353  -0.044273   0.908887   3.37511  -2.54488    4.04813   2.10489  -1.00758   1.38601    2.55543   -1.83667    1.9731    3.61677  -1.51568  -0.117786   1.56233    1.99839   0.51582    0.978938   1.08557   2.19561    5.9363    2.97815    2.27319   1.44133   1.00981   1.40761  0.647008  -0.345412   0.019627  -0.363405  -1.94503   2.7931    2.64808   0.533897  -4.84925    0.508846  -0.623464  -1.52479   1.32458    1.27611  -1.56704   0.292948  -0.0762107  -0.0174109  -3.44954     0.0144578  -3.59336   1.08094  -1.33219    2.00104  -1.11741   0.876026   1.41122   0.803826  -3.03599    0.0551168   0.841822   0.846632   1.99526   -1.9921   -3.9806   -1.71982    1.86339   3.11414   -0.73148     1.74706
````

to define the objective:

````julia
objective = model -> mean_squared_error(model(x), y)

objective(m) # returns some "large" loss value
````

````
18.211070497671898
````

## Gradient computation

Computing the gradient of our objective with respect to the model is easy:

````julia
using Zygote

val, pb = ProtoGrad.eval_with_pullback(objective, m, :Zygote)
grad = pb()
````

````
Main.var"##292".LinearModel
````

Here `val` is the value of the objective evaluated at `m`, while `grad` contains its gradient with respect to **all** attributes of `m`. Most importantly **`grad` is itself a `LinearModel` object**. Therefore, `grad` can be added or subtracted from `m`, used in dot products and so on.

## Fitting models to the objective

Fitting models using gradient-based algorithms is now relatively simple.
The following loop is plain gradient descent with constant stepsize:

````julia
m_fit = copy(m)
for it = 1:100
    val, pb = ProtoGrad.eval_with_pullback(objective, m_fit, :Zygote)
    grad = pb()
    m_fit .= m_fit .- 0.1 .* grad
end
````

````
┌ Warning: Assignment to `val` in soft scope is ambiguous because a global variable by the same name exists: `val` will be treated as a new local. Disambiguate by using `local val` to suppress this warning or `global val` to assign to the existing global variable.
└ @ ~/ProtoGrad.jl/README.md:3
┌ Warning: Assignment to `pb` in soft scope is ambiguous because a global variable by the same name exists: `pb` will be treated as a new local. Disambiguate by using `local pb` to suppress this warning or `global pb` to assign to the existing global variable.
└ @ ~/ProtoGrad.jl/README.md:3
┌ Warning: Assignment to `grad` in soft scope is ambiguous because a global variable by the same name exists: `grad` will be treated as a new local. Disambiguate by using `local grad` to suppress this warning or `global grad` to assign to the existing global variable.
└ @ ~/ProtoGrad.jl/README.md:4

````

To verify that this worked, we can check that the objective value is much smaller for `m_fit` than it was for `m`:

````julia
objective(m_fit) # returns a small loss value compared to `m`
````

````
2.639921626360086
````

ProtoGrad implements gradient descent and other optimization algorithms, with an iterator-like interface:

````julia
using ProtoGrad: Adam, init, step!

optimizer = Adam(; stepsize = 1e-1)
state = init(optimizer, m)
for it = 1:100
    val, pb = ProtoGrad.eval_with_pullback(objective, m, :Zygote)
    grad = pb()
    step!(state, grad)
end

objective(m)
````

````
2.6401329686120683
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

