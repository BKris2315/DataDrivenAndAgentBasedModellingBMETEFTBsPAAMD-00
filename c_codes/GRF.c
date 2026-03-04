#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define GRASS 1

typedef struct animal_ {
  int x,y;
  double E;
} animal;

animal *r,*f;


double ranf() {
  return  (double)rand() / (double)(RAND_MAX+1.0) ;
}

int main(int argc, char **argv) {
  int a,b,i,j,t,N,T,**g,Nr,Nf,Ar,Af,Nr0,Nf0,newNr,newNf;
  double alpha,Er,Ef;

  if (argc<9) {
    fprintf(stderr, "usage: %s N T alpha Nr0 Nf0 Er Ef seed\n",argv[0]);
    exit(0);
  }
  N = atoi(argv[1]);
  T = atoi(argv[2]);
  alpha = atof(argv[3]);
  Nr0 = atof(argv[4]);
  Nf0 = atof(argv[5]);
  Er = atof(argv[6]);
  Ef = atof(argv[7]);
  srand(atoi(argv[8]));
  g = (int **)malloc(N*sizeof(int*));
  for (a=0;a<N;a++) {
    g[a] = calloc(N,sizeof(int));
    for (b=0;b<N;b++) {
      if (ranf() < 0.5) g[a][b]=1;
    }
  }
  Ar = 2*Nr0;
  Af = 2*Nf0;
  r = (animal *)malloc(sizeof(animal)*Ar);
  f = (animal *)malloc(sizeof(animal)*Af);
  Nr = Nr0;
  for (a=0;a<Nr;a++) {
    r[a].x=rand()%N;
    r[a].y=rand()%N;
    r[a].E=1.0;
  }
  Nf = Nf0;
  for (a=0;a<Nf;a++) {
    f[a].x=rand()%N;
    f[a].y=rand()%N;
    f[a].E=1.0;
  }
  fprintf(stderr,"AAA %d %d\n",Nr,Nf);
  for (t=0;t<T;t++) {
    newNr=Nr;
    for (a=0;a<Nr;a++) {
      if (g[r[a].x][r[a].y]==1) {
	g[r[a].x][r[a].y] = 0;
	r[a].E += 1.0;
	if (r[a].E > Er) {
	  r[a].E /= 2.0;
	  if (newNr==Ar) {
	    Ar *= 2;
	    r = (animal *)realloc(r,sizeof(animal)*Ar);
	  }
	  r[newNr].x = r[a].x;
	  r[newNr].y = r[a].y;
	  r[newNr].E = r[a].E;
	  newNr++;
	}
      }
    }
    Nr = newNr;
    for (a=0;a<Nr;a++) {
      r[a].E -= 0.1;
      if (r[a].E <= 0.0) {
	Nr--;
	r[a].x = r[Nr].x;
	r[a].y = r[Nr].y;
	r[a].E = r[Nr].E;
	a--;
      }
    }
    newNf = Nf;
    for (a=0;a<Nf;a++) {
      for (b=0;b<Nr;b++) {
	if ((f[a].x==r[b].x) && (f[a].y==r[b].y)) {
	  f[a].E += r[b].E;
	  Nr--;
	  r[b].x = r[Nr].x;
	  r[b].y = r[Nr].y;
	  r[b].E = r[Nr].E;
	  if (f[a].E > Ef) {
	    f[a].E /= 2.0;
	    if (newNf==Af) {
	      Af *= 2;
	      f = (animal *)realloc(f,sizeof(animal)*Af);
	    }
	    f[newNf].x = f[a].x;
	    f[newNf].y = f[a].y;
	    f[newNf].E = f[a].E;
	    newNf++;
	  }
	}
      }
    }
    Nf = newNf;
    for (a=0;a<Nf;a++) {
      f[a].E -= 0.1;
      if (f[a].E <= 0.0) {
	Nf--;
	f[a].x = f[Nf].x;
	f[a].y = f[Nf].y;
	f[a].E = f[Nf].E;
	a--;
      }
    }
    for (a=0;a<Nr;a++) {
      i = rand()%3-1;
      j = rand()%3-1;
      r[a].x = (r[a].x+i+N)%N;
      r[a].y = (r[a].y+j+N)%N;
    }
    for (a=0;a<Nf;a++) {
      i = rand()%3-1;
      j = rand()%3-1;
      f[a].x = (f[a].x+i+N)%N;
      f[a].y = (f[a].y+j+N)%N;
    }
    for (a=0;a<N;a++) {
      for (b=0;b<N;b++) {
	if (ranf() < alpha) g[a][b]=1;
      }
    }
    fprintf(stderr,"%d %d\n",Nr,Nf);
    printf("%d",N);
    for (a=0;a<N;a++) {
      for (b=0;b<N;b++) {
	printf(" %d",g[a][b]);
      }
    }
    printf(" %d",Nr);
    for (a=0;a<Nr;a++) {
      printf(" %d %d",r[a].x,r[a].y);
    }
    printf(" %d",Nf);
    for (a=0;a<Nf;a++) {
      printf(" %d %d",f[a].x,f[a].y);
    }
    printf("\n");
  }
}
