/****************************************************
 * FileName: set_channel.c
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
    fprintf(stderr,"\n%s n_chan set the voltage to all actuators of AM\n",arr);
}

int main (int argc, char *argv[])
{
	int i, nc;

	if (!init_dac())
		exit(1);

	if (argc != 2)
	{
		printf ("\nThis program will set %d to one channel of OKO mirror.\n"
                "All other channels will be set to 0.\n"
                "The channel number (from 1) should be given as argument\n\n", MAX_AMPLITUDE);
		exit (1);
	}

	sscanf (argv[1], "%d", &nc);

	for (i = 0; i < TOTAL_NR_OF_CHANNELS; i++) voltage[i]=0;
	voltage[nc-1]=MAX_AMPLITUDE;
	set_mirror();
	close_dac();
	return 0;
}
