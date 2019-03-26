import Distributions.pdf

function pdf(x::AbstractVector,z::AbstractVector)
  y=Vector(undef,length(x))
  for i in 1:length(x)
    y[i]=pdf(x[i],z[i])
  end
  return(prod(y))
end
