/****************************************************
 * FileName: smiley37.c
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
	int i;

	if (!init_dac())
		exit(1);

	if (TOTAL_NR_OF_CHANNELS!=19)
	{
		printf("\nThis is not a 19-ch mirror\n");
		exit(0);
	}

	printf ("\nThis program will set smiley to OKO 19ch mirror.\n");

	for (i = 0; i < TOTAL_NR_OF_CHANNELS; i++) voltage[i]=0;
	voltage[1]=MAX_AMPLITUDE;
	voltage[7]=MAX_AMPLITUDE;
	voltage[3]=MAX_AMPLITUDE;
	voltage[11]=MAX_AMPLITUDE;
	voltage[5]=MAX_AMPLITUDE;
	voltage[15]=MAX_AMPLITUDE;
	set_mirror();
	close_dac();

	return 0;
}
