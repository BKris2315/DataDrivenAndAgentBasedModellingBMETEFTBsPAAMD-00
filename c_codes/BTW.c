#include <stdlib.h>
#include <stdio.h>
#include <math.h>

double ranf() {
  return  (double)rand() / (double)(RAND_MAX+1.0) ;
}

int main(int argc, char **argv) {
  int a,b,i,j,t,N,N1,N2,T,*p,*q,qp,avs,snapshot;

  if (argc<3) {
    fprintf(stderr, "usage: %s N T [nosnaphot]\n",argv[0]);
    exit(0);
  }
  N = atoi(argv[1]);
  T = atoi(argv[2]);
  snapshot = 1;
  if (argc>3) snapshot = 0;
  N1 = N-1;
  N2 = N*N;
  p = (int *)calloc(N2,sizeof(int));
  q = (int *)calloc(N2,sizeof(int));
  qp=0;
  avs=0;
  for (t=0;t<T;t++) {
    a=rand()%N2;
    p[a]++;
    if (p[a]>=4) {
      avs=0;
      q[0]=a;
      qp=1;
      avs=0;
      while (qp) {
	avs++;
	a=q[--qp];
	p[a]-=4;
	if (p[a]>=4) q[qp++]=a;
	if (a%N!=0) {
	  p[a-1]++;
	  if (p[a-1]>=4) q[qp++]=a-1;
	}
	if (a%N!=N1) {
	  p[a+1]++;
	  if (p[a+1]>=4) q[qp++]=a+1;
	}
	if (a>=N) {
	  p[a-N]++;
	  if (p[a-N]>=4) q[qp++]=a-N;
	}
	if (a<N2-N) {
	  p[a+N]++;
	  if (p[a+N]>=4) q[qp++]=a+N;
	}
      }
      printf("%d",avs);
      if (snapshot) {
	for (a=0;a<N*N;a++) {
	  printf(" %d",p[a]);
	}
      }
      printf("\n");
    }
  }
}
