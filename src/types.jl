const PopulationMatrix = AbstractMatrix{Float64}
const CovarianceMatrix = AbstractMatrix{Float64}
const WeightsVector = AbstractVector{Float64}
# parameter priors and names: a vector for each model with length = nparameters
const ParameterPriorVector = AbstractVector{ContinuousUnivariateDistribution}
const NamesVector = AbstractVector{String}  # or Symbol?
# is subtyping better?

# Note: priors on models assumed to be uniform for now

struct RejectionInput
  simulators::AbstractVector{Function}
  parameterpriors::AbstractVector{ParameterPriorVector}
  metric::Function
  populationsize::Int
  nkeep::Int
  mindistance::Float64
  names::Union{AbstractVector{NamesVector}, Nothing}
  maxnparams::Int
  nmodels::Int

  function RejectionInput(
      simulators, parameterpriors, metric,
      populationsize, mindistance, names, maxnparams, nmodels)
    if populationsize <= 0
      throw(DomainError(populationsize,
        "population size must be greater than zero"))
    elseif nkeep <= 0
      throw(DomainError(nkeep,
        "number of kept particles must be greater than zero"))
    elseif mindistance <= 0.0
      throw(DomainError(mindistance,
        "minimum accepted distance must be greater than zero"))
    elseif maxnparams != maximum(length.(parameterpriors))
      throw(DomainError(maxnparams,
        "supplied maxnparams must agree with calculated value"))
    elseif nmodels != length(simulators) || nmodels != length(parameterpriors)
      throw(DomainError(nmodels,
        "supplied nmodels must agree with calculated value"))
    end
    return new(
      simulators, parameterpriors, metric,
      populationsize, nkeep, mindistance, names, maxnparams, nmodels
      )
  end
end

function RejectionInput(
    simulators, parameterpriors, metric;
    populationsize = 1000,
    quantilethreshold = 0.5,
    mindistance = Inf,
    names = nothing
    )
  nkeep = round(Int, quantilethreshold * populationsize)
  if names === nothing
    names = [
      [string("p", i) for i in eachindex(parameterpriors[m])]
      for m in eachindex(simulators)
      ]
  end
  maxnparams = maximum(length.(parameterpriors))
  nmodels = length(simulators)
  return RejectionInput(
    simulators, parameterpriors, metric,
    populationsize, nkeep, mindistance, names, maxnparams, nmodels
    )
end

struct APMCInput
  simulators::AbstractVector{Function}
  parameterpriors::AbstractVector{ParameterPriorVector}
  metric::Function
  populationsize::Int
  nkeep::Int
  minacceptance::Float64
  names::Union{AbstractVector{NamesVector}, Nothing}

  function APMCInput(
      simulators, parameterpriors, metric,
      populationsize, quantilethreshold, minacceptance, names)
    if populationsize <= 0
      throw(DomainError(populationsize,
        "population size must be greater than zero"))
    elseif nkeep <= 0
      throw(DomainError(nkeep,
        "number of kept particles must be greater than zero"))
    elseif !(0.0 <= minacceptance <= 1.0)
      throw(DomainError(minacceptance,
        "minimum acceptance rate must be between zero and one"))
    end
    return new(
      simulators, parameterpriors, metric,
      populationsize, nkeep, minacceptance, names
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
  nkeep = round(Int, quantilethreshold * populationsize)
  if names === nothing
    names = [
      [string("p", i) for i in eachindex(parameterpriors[m])]
      for m in eachindex(simulators)
      ]
  end
  return APMCInput(
    simulators, parameterpriors, metric,
    populationsize, nkeep, minacceptance, names
    )
end

# APMC algorithm output structure
struct ModelSelectionResult
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
  latest_population::AbstractMatrix{Float64}  # rewrite in terms of new Particle type
  latest_distances::AbstractVector{Float64}
  # Information about the progression of the algorithm: one item per iteration
  ntries::AbstractVector{Int}
  epsilons::AbstractVector{Float64}
  # Acceptance rates per iteration i and model j
  acceptance_rates::AbstractMatrix{Float64}
  # Priors and names of variables used--one list per model
  parameterpriors::AbstractVector{ParameterPriorVector}
  names::AbstractVector{NamesVector}
  # To Do: must include enough information to restart simulation.
end

# abstract type ModelSelectionParticle <: Array{Float64}

struct Particle
  model_index::Int
  parameters::Vector{Float64}  # use StaticArrays?
  distance::Float64
end
EmptyParticle() = Particle(0, [], 0.0)

# To Do: write concatenation function for ModelSelectionResults
