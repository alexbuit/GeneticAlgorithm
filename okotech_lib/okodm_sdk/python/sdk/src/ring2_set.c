/****************************************************
 * FileName: ring2_set.c
 * Description:
 *
 * (c) 2004, FlexibleOptical
 *           Gleb Vdovin & Mikhail Loktev
 ****************************************************
 */

#include <stdio.h>
#include "mirror_edac40.h"

int main (int argc, char *argv[])
{
	int i, nc;
	char mess[255];
	int first_channel;

	if (argc != 2)
		nc = MAX_AMPLITUDE;
	else
		sscanf (argv[1], "%d", &nc);

	if (!init_dac())
		exit(1);

    if(TOTAL_NR_OF_CHANNELS==39)         first_channel=15;
      else if(TOTAL_NR_OF_CHANNELS==79)  first_channel=49;
        else if(TOTAL_NR_OF_CHANNELS==19)  first_channel=6;
            else
          {
             printf("\nThis is neither 19-ch nor a 39-ch nor a 79-ch mirror\n");
		     exit(0);
          }


	sprintf(mess, "\nThis program will set %d to %d-%d channels of OKO %dch mirror.\n\n",
            nc,first_channel,TOTAL_NR_OF_CHANNELS,TOTAL_NR_OF_CHANNELS);
	printf(mess);

	for (i=0; i<first_channel; i++) voltage[i]=0;
	for (i=first_channel; i<TOTAL_NR_OF_CHANNELS; i++) voltage[i]=nc;
	set_mirror();
	close_dac();
	return 0;
}

