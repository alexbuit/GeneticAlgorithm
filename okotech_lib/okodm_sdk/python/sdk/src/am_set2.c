/****************************************************
 * FileName: am_set.c
 * Description:
 *
 * (c) 2004, FlexibleOptical
 *           Gleb Vdovin & Mikhail Loktev
 ****************************************************
 */

#include <stdio.h>
#include "mirror_edac40.h"

void error_print(char *arr)
{
    fprintf(stderr,"\n%s  set the voltage to all actuators of AM\n",arr);
}


int main (int argc, char *argv[])
{

	unsigned int V,i;

	if (!init_dac())
		exit(1);

	if (argc != 2 )
	{
		error_print(argv[0]);
		exit(1);
    }

	sscanf(argv[1],"%u", &V);

	for(i=0; i<TOTAL_NR_OF_CHANNELS-2; i++) voltage[i] = V;
	voltage[TOTAL_NR_OF_CHANNELS-2] = voltage[TOTAL_NR_OF_CHANNELS-1] = 0;
	set_mirror();
	close_dac();
	return 0;
}

