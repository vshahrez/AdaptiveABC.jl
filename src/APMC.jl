#rejection samplerm (for first iteration)
function init(models,expd,np,rho; mineps = Inf)
  worker = myid()
  d=Inf
  count = 0
  while d >= mineps
    count += 1
    m=sample(1:length(models))
    params=rand(models[m])
    d=rho[m](expd,params)
    if d < mineps
      return vcat(m,params,fill(0,maximum(np)-np[m]),d,count)
    end
    if count % 100 == 0
      @warn ">100 particles rejected due to mineps" count worker
    end
  end
end

#SMC sampler (for subsequent iterations)
function cont(models,pts,wts,expd,np,i,ker,rho)
  d=Inf
  count=1
  #while d==Inf
  m=sample(1:length(models))
  while size(pts[m,i-1])[2]==0
    m=sample(1:length(models))
  end
  params=pts[m,i-1][:,sample(1:size(pts[m,i-1])[2],wts[m,i-1])]
  params=params+rand(ker[m])
  while pdf(models[m],params)==0
    m=sample(1:length(models))
    while size(pts[m,i-1])[2]==0
      m=sample(1:length(models))
    end
    count=count+1
    params=pts[m,i-1][:,sample(1:size(pts[m,i-1])[2],wts[m,i-1])]
    params=params+rand(ker[m])
  end
  d=rho[m](expd,params)
  # end
  return vcat(m,params,fill(0,maximum(np)-np[m]),d,count)
end

function APMC(N,expd,models,rho,;names=Vector[[string("parameter",i) for i in 1:length(models[m])] for m in 1:length(models)],prop=0.5,paccmin=0.02,n=2, mineps = Inf)
  i=1
  lm=length(models)
  s=round(Int,N*prop)
  #array for number of parameters in each model
  np=Array{Int64}(undef,length(models))
  for j in 1:lm
    np[j]=length(models[j])
  end
  #array for SMC kernel used in weights
  ker=Array{Any}(undef,lm)
  template=Array{Any}(undef,lm,1)
  #particles array
  pts=similar(template)
  #covariance matrix array
  sig=similar(template)
  #weights array
  wts=similar(template)
  #model probability at each iteration array
  p=zeros(lm,1)
  temp=@distributed hcat for j in 1:N
    init(models,expd,np,rho; mineps = mineps)
  end
  its=[sum(temp[size(temp)[1],:])]
  epsilon=[quantile(collect(temp[maximum(np)+2,:]),prop)]
  pacc=ones(lm,1)
  println(round.([epsilon[i];its[i]],digits=3))
  temp=temp[:,temp[maximum(np)+2,:].<=epsilon[i]]
  temp=temp[:,1:s]
  for j in 1:lm
    pts[j,i]=temp[2:(np[j]+1),temp[1,:].==j]
    wts[j,i]=weights(fill(1.0,sum(temp[1,:].==j)))
  end
  dists=transpose(temp[(maximum(np)+2),:])
  for j in 1:lm
    p[j]=sum(wts[j,1])
  end
  for j in 1:lm
    sig[j,i]=cov(pts[j,i],wts[j,i],2,corrected=false)
  end
  p=p./sum(p)
  nbs=Array{Integer}(undef,length(models))
  for j in 1:lm
    nbs[j]=length(wts[j,i])
    println(round.(hcat(mean(diag(sig[j,i])[1:(np[j])]),pacc[j,i],nbs[j],p[j,i]),digits=3))
  end
  flush(stdout)
  while maximum(pacc[:,i])>paccmin
    pts=reshape(pts,i*length(models))
    sig=reshape(sig,i*length(models))
    wts=reshape(wts,i*length(models))
    for j in 1:length(models)
      push!(pts,Array{Any}(undef,1))
      push!(sig,Array{Any}(undef,1))
      push!(wts,Array{Any}(undef,1))
    end
    pts=reshape(pts,length(models),i+1)
    sig=reshape(sig,length(models),i+1)
    wts=reshape(wts,length(models),i+1)
    i=i+1
    for j in 1:lm
      ker[j]=MvNormal(fill(0.0,np[j]),n*sig[j,i-1])
    end
    temp2=@distributed hcat for j in (1:(N-s))
      cont(models,pts,wts,expd,np,i,ker,rho)
    end
    its=vcat(its,sum(temp2[size(temp2)[1],:]))
    temp=hcat(temp,temp2)
    inds=sortperm(reshape(temp[maximum(np)+2,:],N))[1:s]
    temp=temp[:,inds]
    dists=hcat(dists,transpose(temp[(maximum(np)+2),:]))
    epsilon=vcat(epsilon,temp[(maximum(np)+2),s])
    pacc=hcat(pacc,zeros(lm))
    for j in 1:lm
        if sum(temp2[1,:].==j)>0
      pacc[j,i]=sum(temp[1,inds.>s].==j)/sum(temp2[1,:].==j)
      else pacc[j,i]==0
      end
    end
    println(round.(vcat(epsilon[i],its[i]),digits=3))
    for j in 1:lm
      pts[j,i]=temp[2:(np[j]+1),temp[1,:].==j]
      if size(pts[j,i])[2]>0
        keep=inds[reshape(temp[1,:].==j,s)].<=s
        wts[j,i]= @distributed vcat for k in 1:length(keep)
          if !keep[k]
            pdf(models[j],(pts[j,i][:,k]))/(1/(sum(wts[j,i-1]))*dot(values(wts[j,i-1]),pdf(ker[j],broadcast(-,pts[j,i-1],pts[j,i][:,k]))))
          else
            0.0
          end
        end
        if length(wts[j,i])==1
          wts[j,i]=fill(wts[j,i],1)
        end
        l=1
        for k in 1:length(keep)
          if keep[k]
            wts[j,i][k]=wts[j,i-1][l]
            l=l+1
          end
        end
          if length(wts[j,i])>1
        wts[j,i]=weights(wts[j,i])
          end
      else
        wts[j,i]=zeros(0)
      end
    end
    p=hcat(p,zeros(length(models)))
    for j in 1:lm
      p[j,i]=sum(wts[j,i])
    end
    for j in 1:lm
      if(size(pts[j,i])[2]>np[j])
        #sig[j,i]=cov(transpose(pts[j,i]),wts[j,i])
        sig[j,i]=cov(pts[j,i],wts[j,i],2,corrected=false)
        if isposdef(sig[j,i])
          dker=MvNormal(pts[j,i-1][:,1],n*sig[j,i])
          if pdf(dker,pts[j,i][:,1])==Inf
            sig[j,i]=sig[j,i-1]
          end
        else
          sig[j,i]=sig[j,i-1]
        end
      else
        sig[j,i]=sig[j,i-1]
      end
    end
    p[:,i]=p[:,i]./sum(p[:,i])
    for j in 1:lm
      nbs[j]=length(wts[j,i])
      println(round.(hcat(mean(diag(sig[j,i])./diag(sig[j,1])),pacc[j,i],nbs[j],p[j,i]),digits=3))
    end
    flush(stdout)
  end
  @Base.CoreLogging.logmsg(Base.LogLevel(1100),
    "APMC result incoming",
    pts, sig, wts, p, temp, dists, its, epsilon, pacc, models, names
    )
  samp=APMCResult(pts,sig,wts,p,temp,vec(dists),its,epsilon,pacc,models,names)
  return(samp)
end
