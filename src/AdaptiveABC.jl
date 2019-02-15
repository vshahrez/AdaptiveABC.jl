module AdaptiveABC

using Distributions
using LinearAlgebra: dot
using StatsBase: weights

include("types.jl")
include("APMC.jl")

export ABCfit,
  APMC

end # module
