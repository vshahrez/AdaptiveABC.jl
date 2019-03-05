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
  particles = @distributed vcat for j in 1:input.populationsize  # previously hcat of vectors
    rejection_sample(input, reference)  # previously init(models,expd,np,rho)
  end
  ntries = sum(particle.ntries for particle in particles)

  # Sort particle vector and keep only the top
  getdistance(p) = p.distance
  sort!(particles, by = getdistance)
  particles = particles[1:input.nkeep]
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
  for j in 1:index.nmodels
    nparticles = length(particles_by_model[j])
    populations[1, j] = [
      particles_by_model[j][k][l].parameters
      for l in 1:nparticles,
        k in 1:input.nparameters[j]
      ]
    weights[1, j] = weights(ones(input.nparameters[j], nparticles))
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

function rejection_sample(
    input::RejectionInput,
    reference
    )
  distance = Inf
  ntries = 0
  while distance >= input.mindistance
    model_index = sample(1:input.nmodels)
    parameters = rand.(input.priors[model_index])
    generated_data = input.simulators[model_index](parameters)  # include in output somehow?
    ntries += 1
    distance = input.metric(generated_data, reference)
  end
  return RejectionParticle(model_index, parameters, distance, ntries)
end
