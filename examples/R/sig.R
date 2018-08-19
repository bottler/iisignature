#Takes a path in R^d given by an array of n points with dims c(d,n)
#and returns its signature up to level m excluding level 0
#e.g. path=array(c(1,1,3,10,20,30),c(3,2))
#Note that the order of path's dimensions is the other 
#way around from iisignature.sig.
sig=function(path,m, flat=TRUE){
  n=dim(path)[2]
  d=dim(path)[1]
  diffs=path[,-1,drop=FALSE]-path[,-n,drop=FALSE]
  if (n<2){
   o=sapply(1:m,function(x)rep(0,d**x))
   if (flat) return (do.call(c,o))
   return (o)
  }
  r=lapply(1:(n-1),function(x)Reduce(kronecker,rep(list(diffs[,x]),m),accumulate=TRUE))
  facts=lapply(1:m,factorial)
  r=lapply(r,function(x)mapply("/",x,facts))
  chen=function(x,y) c(list(x[[1]]+y[[1]]),
                       lapply(2:m,
                              function(z)x[[z]]+y[[z]]
                               +Reduce("+",mapply(kronecker,x[1:z-1],rev(y[1:z-1]),
                                                  SIMPLIFY=FALSE))))
  o=Reduce(chen,r)
  if (flat) return (do.call(c,o))
  o
}

