/**************************************************** 
 * FileName: degauss.cpp
 * Description:  
 *
 * (c) 2006, FlexibleOptical 
 *           Gleb Vdovin     
 ****************************************************
 */

#include "mirror.h"

int
main (int argc, char *argv[])
{
	int i, ii,  j, jj;

	if (!init_dac())
		exit(1);

	printf ("\n\
This program will set 255V to  all  channels  of OKO 37ch mirror periodically \n");


	//  sscanf (argv[1], "%d", &nc);
	for (i=0; i<37; i++) voltage[i] =0.;
  	set_mirror();
	for (jj=0; jj< 250000; jj++);

	for (i=0; i<37; i++) 
	{
		printf("Channel =  %d \n", i+1);
		for (j=4090; j>=0; j-=10)
		{
			for(ii=1; ii<=1; ii++)
			{
				voltage[i] = 0;
				set_mirror();
				for (jj=0; jj< 2500; jj++);
				voltage[i] = j;
				set_mirror();
				for (jj=0; jj< 2500; jj++);
			}
		}
	}
	close_dac();
}
