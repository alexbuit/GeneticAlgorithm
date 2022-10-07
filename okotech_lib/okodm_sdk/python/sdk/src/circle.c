#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mirror_edac40.h"

int main (int argc, char *argv[])
{
	//const double M_PI = 4*atan(1.0);

	int sleep4 = 1000000, nc, i,j;
	//char mess[255];

	if(argc == 2)
	{
		sscanf(argv[1], "%d", &sleep4);
	}
	
	
	//if (argc != 2)
	//	nc = 255;
	//else
	//	sscanf (argv[1], "%d", &nc);

	if (!init_dac())
		exit(1);

	for( i = 0; i < TOTAL_NR_OF_CHANNELS; i++) 
		voltage[i]=0;
	set_mirror();
	Sleep(500);

	nc = MAX_AMPLITUDE;
	for( i = 0; ; i++ )
	{
		voltage[17] = nc/2*(1+sin(2*M_PI*i/140.));
		voltage[18] = nc/2*(1+cos(2*M_PI*i/140.));
		set_mirror();
		//Sleep(sleep4);
		for( j=1; j< sleep4; j++);
	}

	close_dac();
	
	return 0;
}
