module AdaptiveABC
  using Distributed: @distributed
  using Distributions
  using LinearAlgebra
  using StatsBase: weights

  include("types.jl")
  include("utils.jl")
  include("APMC.jl")
  include("modelselection.jl")

  export APMCInput, APMCResult,
    APMC,
    modelselection

end # module
