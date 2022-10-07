/**************************************************** 
 * FileName: am_set.cpp
 * Description:  
 *
 * (c) 2004, FlexibleOptical
 *           Gleb Vdovin & Mikhail Loktev
 ****************************************************
 */

#include "mirror.h"

void error_print(char *arr) 
{
    fprintf(stderr,"\n%s  set the voltage\
to all actuators of AM\n",arr);
}


int main (int argc, char *argv[])
{

	int V,i;

	if (!init_dac())
		exit(1);

	if (argc != 2 )
	{
		error_print(argv[0]);
		exit(1);
    }

	sscanf(argv[1],"%d", &V);

	for(i=0; i<TOTAL_NR_OF_CHANNELS; i++) voltage[i] = V;
	set_mirror();
	close_dac();
	return 0;
}
