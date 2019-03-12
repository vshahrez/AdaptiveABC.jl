module AdaptiveABC
  using Distributed
  using Distributions
  using LinearAlgebra
  using Statistics
  using StatsBase: weights

  include("types.jl")
  include("utils.jl")
  include("APMC.jl")
  include("modelselection.jl")

  export APMCInput, APMCResult,
    APMC,
    modelselection

end # module
