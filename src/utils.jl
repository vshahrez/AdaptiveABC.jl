import Distributions.rand
import Distributions.pdf

function rand(x::AbstractVector)
  y=zeros(length(x))
  for i in 1:length(x)
    y[i]=rand(x[i])
  end
  return(y)
end

function pdf(x::AbstractVector,z::AbstractVector)
  y=Vector(undef,length(x))
  for i in 1:length(x)
    y[i]=pdf(x[i],z[i])
  end
  return(prod(y))
end
