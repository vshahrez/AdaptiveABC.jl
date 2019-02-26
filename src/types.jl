const PopulationMatrix = AbstractMatrix{Float64}
const CovarianceMatrix = AbstractMatrix{Float64}
const WeightsVector = AbstractVector{Float64}
# parameter priors and names: a vector for each model with length = nparameters
const ParameterPriorVector = AbstractVector{ContinuousUnivariateDistribution}
const NamesVector = AbstractVector{String}  # or Symbol?

# Note: priors on models assumed to be uniform for now

struct RejectionInput
  simulators::AbstractVector{Function}
  parameterpriors::AbstractVector{ParameterPriorVector}
  metric::Function
  populationsize::Int
  mindistance::Float64
  names::Union{AbstractVector{NamesVector}, Nothing}
  maxnparams::Int

  function RejectionInput(
      simulators, parameterpriors, metric,
      populationsize, mindistance, names, maxnparams)
    if populationsize <= 0
      throw(DomainError(populationsize,
        "population size must be greater than zero"))
    elseif mindistance <= 0.0
      throw(DomainError(mindistance,
        "minimum accepted distance must be greater than zero"))
    elseif maxnparams != maximum(length.(parameterpriors))
      throw(DomainError(maxnparams,
        "supplied maxnparams must agree with calculated value"))
    end
    return new(
      simulators, parameterpriors, metric,
      populationsize, mindistance, names, maxnparams
      )
  end
end

function RejectionInput(
    simulators, parameterpriors, metric;
    populationsize = 1000,
    mindistance = Inf,
    names = nothing
    )
  if names === nothing
    names = [
      [string("p", i) for i in eachindex(parameterpriors[m])]
      for m in eachindex(simulators)
      ]
  end
  maxnparams = maximum(length.(parameterpriors))
  return RejectionInput(
    simulators, parameterpriors, metric,
    populationsize, mindistance, names, maxnparams
    )
end

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
    if populationsize <= 0
      throw(DomainError(populationsize,
        "population size must be greater than zero"))
    elseif !(0.0 <= quantilethreshold <= 1.0)
      throw(DomainError(quantilethreshold,
        "quantile threshold must be between zero and one"))
    elseif !(0.0 <= minacceptance <= 1.0)
      throw(DomainError(minacceptance,
        "minimum acceptance rate must be between zero and one"))
    end
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
  # they squash all models together
  # and have extra entries for the model index (row 1),
  # distance to reference (row n + 2)
  # and particle weight (row n + 3)
  latest_population::AbstractMatrix{Float64}
  latest_distances::AbstractVector{Float64}
  # Information about the progression of the algorithm: one item per iteration
  ntries::AbstractVector{Int}
  epsilons::AbstractVector{Float64}
  # Acceptance rates per iteration i and model j
  acceptance_rates::AbstractMatrix{Float64}
  # Priors and names of variables used--one list per model
  parameterpriors::AbstractVector{ParameterPriorVector}
  names::AbstractVector{NamesVector}
end
