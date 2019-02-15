module AdaptiveABC
  using Distributed: @distributed
  using Distributions
  using LinearAlgebra: dot
  using StatsBase: weights

  include("types.jl")
  include("APMC.jl")

  export APMCInput, APMCResult,
    modelselection, APMC

end # module
