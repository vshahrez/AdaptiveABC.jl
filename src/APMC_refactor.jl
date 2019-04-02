function modelselection(
    input::APMCInput,
    reference
    )
  # initial setup: run rejection sampler to generate first population
  rejection_input = RejectionInput(input)  # extend Base.convert?
  rejection_result = modelselection(rejection_input, reference)

  # then, keep performing PMC sampling until stoppage criterion met
  apmc_result = modelselection(rejection_result, reference)  # alg = APMC() ?

  return apmc_result
end

function modelselection(
    input::RejectionInput,
    reference
    )
  # initial setup?

  # core computation: distributed for loop
  iterations = @distributed vcat for j in 1:input.populationsize  # previously hcat of vectors
    particle = EmptyParticle()
    ntries = 0
    while particle == EmptyParticle()
      particle = rejection_sample(input, reference)  # previously init(models,expd,np,rho)
      ntries += 1
    end
    (particle = particle, ntries = ntries)
  end
  ntries = sum(iteration.ntries for iteration in iterations)

  # Sort particle vector and keep only the top
  getdistance(iteration) = iteration.particle.distance
  sort!(iterations, by = getdistance)
  particles = [iteration.particle for iteration in iterations[1:input.nkeep]]
  epsilon = particles[end].distance

  # filter particles according to the model that generated them
  particle_in_model(j) = particle -> particle.model_index == j
  particles_by_model = [filter(particle_in_model(j), particles) for j in 1:input.nmodels]

  # generate the output
  populations = Matrix{Matrix{Float64}}(undef, 1, input.nmodels)
  covariances = similar(populations)
  weights = Matrix{Vector{Float64}}(undef, 1, input.nmodels)
  probabilities = Matrix{Vector{Float64}}(undef, 1, input.nmodels)
  # iteration i = 1
  for j in 1:input.nmodels
    nparticles = length(particles_by_model[j])
    populations[1, j] = [
      particles_by_model[j][k][l].parameters
      for l in 1:nparticles,
        k in 1:input.nparameters[j]
      ]
    weights[1, j] = StatsBase::weights(ones(input.nparameters[j], nparticles))
    covariances[1, j] = cov(populations[1, j], weights[1, j], vardim = 2, corrected = false)
    probabilities[1, j] = sum(weights[1, j])  # requires normalisation!?
  end
  temp_distances = [p.distance for p in particles]
  temp_population = vcat(
    [p.model_index for p in particles],
    [p.parameters for p in particles],
    temp_distances,
    [1.0 for p in particles]  # weights
    )
  ntries = sum(particle.ntries for particle in particles)
  epsilon = particles[end].distance
  acceptance_rates = ones(1, input.nmodels)

  return ModelSelectionResult(
    populations,
    covariances,
    weights,
    probabilities,
    temp_population,
    temp_distances,
    [ntries],
    [epsilon],
    acceptance_rates,
    input.parameterpriors,
    input.names
    )
end

function modelselection(
    input::ModelSelectionResult,
    reference
    )
  # To Do: Add alg keyword (using singleton type?) to indicate usage of APMC
  #        handle options for choosing the perturbation kernels
  #        refactor code to generalise from modelselection(::RejectionInput)

  # setup perturbation kernels
  kernels = generate_kernels(input)

  iterations = @distributed vcat for j in 1:input.populationsize
    particle = EmptyParticle()
    ntries = 0
    while particle == EmptyParticle()
      particle = pmc_sample(input, reference, kernels)
      ntries += 1
    end
    (particle = particle, ntries = ntries)
  end
  ntries = sum(iteration.ntries for iteration in iterations)

  # Sort particle vector and keep only the top
  getdistance(iteration) = iteration.particle.distance
  sort!(iterations, by = getdistance)
  particles = [iteration.particle for iteration in iterations[1:input.nkeep]]
  # Merge with previous APMC run to obtain threshold
  epsilon = ...
  # Re-sort?

  # filter particles according to the model that generated them
  particle_in_model(j) = particle -> particle.model_index == j
  particles_by_model = [filter(particle_in_model(j), particles) for j in 1:input.nmodels]

  # generate the output
  populations = Matrix{Matrix{Float64}}(undef, 1, input.nmodels)
  covariances = similar(populations)
  weights = Matrix{Vector{Float64}}(undef, 1, input.nmodels)
  probabilities = Matrix{Vector{Float64}}(undef, 1, input.nmodels)
  for j in 1:input.nmodels
    nparticles = length(particles_by_model[j])
    populations[1, j] = [
      particles_by_model[j][k][l].parameters
      for l in 1:nparticles,
        k in 1:input.nparameters[j]
      ]
    weights[1, j] = ... # NORMALISATION
    covariances[1, j] = cov(populations[1, j], weights[1, j], vardim = 2, corrected = false)
    # Check conditions on covariance matrix for models with few particles accepted
    probabilities[1, j] = sum(weights[1, j])
  end
  temp_distances = [p.distance for p in particles]
  temp_population = vcat(
    [p.model_index for p in particles],
    [p.parameters for p in particles],
    temp_distances,
    [1.0 for p in particles]  # weights  - CHANGE TO NORMALISED WEIGHTS
    )
  ntries = sum(particle.ntries for particle in particles)
  epsilon = particles[end].distance
  acceptance_rates = ... # CALCULATE

  new_result = ModelSelectionResult(
    populations,
    covariances,
    weights,
    probabilities,
    temp_population,
    temp_distances,
    [ntries],
    [epsilon],
    acceptance_rates,
    input.parameterpriors,
    input.names
    )
  result = concatenate(input, new_result)
  # if stoppage criterion not yet attained, recursively call modelselection()
  if any(result.acceptance_rates[end] .> input.minacceptance)  # minacceptance not in input
    modelselection(result, reference)
  end
  return result
end

function rejection_sample(
    input::RejectionInput,
    reference
    )
  model_index = sample(eachindex(input.parameterpriors))
  parameters = rand.(input.parameterpriors[model_index])
  generated_data = input.simulators[model_index](parameters)  # include in output somehow?
  distance = input.metric(generated_data, reference)

  if distance < input.mindistance
    return Particle(model_index, parameters, distance)
  else
    return EmptyParticle()
  end
end

function pmc_sample(
    input::ModelSelectionResult,
    reference,
    perturbation_kernels
    )
  model_index = sample(eachindex(input.parameterpriors))
  if size(input.populations[end, model_index], 2) == 0  # model is dead
    return EmptyParticle()
  end
  # sample particle from previous population using previous weights
  parameters = sample(input.populations[end, model_index], input.weights[end, model_index])
  # then perturb particle with PMC kernel
  parameters += rand(perturbation_kernels[model_index])
  # if perturbed particle outside the prior
  if prod(pdf.(input.parameterpriors, parameters)) == 0.0
    return EmptyParticle()
  end

  generated_data = input.simulators[model_index](parameters)
  distance = input.metric(generated_data, reference)

  return Particle(model_index, parameters, distance)
end

function generate_kernels(input::ModelSelectionResult)
  nmodels = length(input.parameterpriors)
  means = [zeros(length(input.parameterpriors[j]) for j in 1:nmodels]
  covariances = [covariances[end, j] for j in 1:nmodels]

  return [MvNormal(means[j], 2.0 * covariances[j]) for j in 1:nmodels]
end
