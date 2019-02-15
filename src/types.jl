# ABC algorithm output structure
struct APMCResult
  # for these four, M[i, j] corresponds to iteration i and model j
  populations::AbstractMatrix{PopulationMatrix}
  covariances::AbstractMatrix{CovarianceMatrix}
  weights::AbstractMatrix{WeightsVector}
  probabilities::AbstractMatrix{Float64}
  # these two correspond to the latest iteration only
  # they squash all models together (and have an extra model indexing parameter)
  latest_population::AbstractMatrix{Float64}
  latest_distances::AbstractVector{Float64}
  # Information about the progression of the algorithm: one item per iteration
  ntries::AbstractVector{Int}
  epsilons::AbstractVector{Float64}
  acceptance_rates::AbstractVector{Float64}
  # Names of variables used--one list per model
  names::AbstractVector{NamesVector}
end

const PopulationMatrix = AbstractMatrix{Float64}
const CovarianceMatrix = AbstractMatrix{Float64}
const WeightsVector = AbstractVector{Float64}
const NamesVector = AbstractVector{String}  # or Symbol?
