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
  temp_population = @distributed hcat for j in 1:input.populationsize
    rejection_sample(input, reference)  # previously init(models,expd,np,rho)
  end
end

function rejection_sample(
    input::RejectionInput,
    reference
    )
  distance = Inf
  ntries = 0
  while distance >= input.mindistance
    model_index = sample(1:length(input.simulators))  # store the number of models explicitly?
    parameters = rand.(input.priors[model_index])
    generated_data = input.simulators[model_index](parameters)
    ntries += 1
    distance = input.metric(generated_data, reference)
  end
  nzeros = input.maxnparams - length(input.parameterpriors[model_index])
  pad_zeros = fill(0.0, nzeros)
  return vcat(model_index, parameters, pad_zeros, ntries)
end
