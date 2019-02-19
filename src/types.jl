const ParameterPrior = AbstractVector{ContinuousUnivariateDistribution}  # multivariate?
# Note: priors on models assumed to be uniform for now

const PopulationMatrix = AbstractMatrix{Float64}
const CovarianceMatrix = AbstractMatrix{Float64}
const WeightsVector = AbstractVector{Float64}
const ParameterPriorVector = AbstractVector{ParameterPrior}
const NamesVector = AbstractVector{String}  # or Symbol?

struct APMCInput
  simulators::AbstractVector{Function}
  parameterpriors::AbstractVector{ParameterPriorVector}
  metric::Function
  populationsize::Int
  quantilethreshold::Float64
  minacceptance::Float64
  names::Union{AbstractVector{NamesVector}, Nothing}

  function APMCInput(
      simulators, parameterpriors, metric,
      populationsize, quantilethreshold, minacceptance, names)
    populationsize > 0 || DomainError("population size must be greater than zero")
    0.0 <= quantilethreshold <= 1.0 || DomainError("quantile threshold must be between zero and one inclusive")
    0.0 <= minacceptance <= 1.0 || DomainError("minimum acceptance rate must be between zero and one inclusive")
    return new(
      simulators, parameterpriors, metric,
      populationsize, quantilethreshold, minacceptance, names
      )
  end
end

function APMCInput(
    simulators, parameterpriors, metric;
    populationsize = 1000,
    quantilethreshold = 0.5,
    minacceptance = 0.02,
    names = nothing
    )
  if names === nothing
    names = [
      [string("p", i) for i in eachindex(parameterpriors[m])]
      for m in eachindex(simulators)
      ]
  end
  return APMCInput(
    simulators, parameterpriors, metric,
    populationsize, quantilethreshold, minacceptance, names
    )
end

# APMC algorithm output structure
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
  # Priors and names of variables used--one list per model
  parameterpriors::AbstractVector{ParameterPriorVector}
  names::AbstractVector{NamesVector}
end
