module AdaptiveABC
  using Distributed: @distributed
  using Distributions
  using LinearAlgebra
  using StatsBase: weights

  include("types.jl")
  include("utils.jl")
  include("APMC.jl")

  export APMCInput, APMCResult,
    modelselection, APMC

end # module
