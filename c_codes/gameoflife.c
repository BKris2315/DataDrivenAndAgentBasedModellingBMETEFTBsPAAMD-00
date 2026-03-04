#include <stdlib.h>
#include <stdio.h>
#include <math.h>

double ranf() {
  return  (double)rand() / (double)(RAND_MAX+1.0) ;
}

int main(int argc, char **argv) {
  int a,b,i,j,t,N,T,**p,**r,nn,**d1,**d2,**tmp;
  double rho0;

  if (argc<4) {
    fprintf(stderr, "usage: %s N T rho0\n",argv[0]);
    exit(0);
  }
  N = atoi(argv[1]);
  T = atoi(argv[2]);
  rho0 = atof(argv[3]);
  d1 = (int **)malloc(N*sizeof(int*));
  d2 = (int **)malloc(N*sizeof(int*));
  for (a=0;a<N;a++) {
    d1[a] = calloc(N,sizeof(int));
    d2[a] = calloc(N,sizeof(int));
  }
  for (a=0;a<N;a++) {
    for (b=0;b<N;b++) {
      if (ranf() < rho0) d1[a][b] = 1;
    }
  }
  p=d1;
  r=d2;
  for (t=0;t<T;t++) {
    for (a=0;a<N;a++) {
      for (b=0;b<N;b++) {
	nn=-p[a][b];
	for(i=-1;i<2;i++) {
	  for(j=-1;j<2;j++) {
	    nn+=p[(a+i+N)%N][(b+j+N)%N];
	  }
	}
	if (nn==3) r[a][b]=1;
	else if (nn==2) r[a][b]=p[a][b];
	else r[a][b]=0;
      }
    }
    for (a=0;a<N;a++) {
      for (b=0;b<N;b++) {
	printf("%d ",r[a][b]);
      }
    }
    printf("\n");
    tmp=r;
    r=p;
    p=tmp;
  }
}
