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

	if (TOTAL_NR_OF_CHANNELS!=37)
	{
		printf("\nThis is not a 37-ch mirror\n");
		exit(0);
	}

	printf ("\nThis program will set smiley to OKO 37ch mirror.\n");

	for (i = 0; i < TOTAL_NR_OF_CHANNELS; i++) voltage[i]=0;
	voltage[36]=MAX_AMPLITUDE;
	voltage[19]=MAX_AMPLITUDE;
	voltage[20]=MAX_AMPLITUDE;
	voltage [27]=MAX_AMPLITUDE;
	voltage[26]=MAX_AMPLITUDE;
	voltage[31]=MAX_AMPLITUDE;
	voltage[32]=MAX_AMPLITUDE;
	set_mirror();
	close_dac();

	return 0;
}
