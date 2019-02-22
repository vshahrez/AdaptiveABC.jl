function modelselection(
    input::APMCInput,
    reference;
    APMC_kwargs...
    )
  distance_generators = [
    function(reference_data, parameters)
      simulated_data = simulator(parameters)
      return input.metric(simulated_data, reference_data)
    end
    for simulator in input.simulators
    ]
  result = APMC(
    input.populationsize,
    reference,
    input.parameterpriors,
    distance_generators;
    names = input.names,
    prop = input.quantilethreshold,
    paccmin = input.minacceptance,
    APMC_kwargs...
    )

  return result
end
