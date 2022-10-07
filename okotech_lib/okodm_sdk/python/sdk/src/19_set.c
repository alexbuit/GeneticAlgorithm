/****************************************************
 * FileName: 19_set.c
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

	if (argc != 2)
		nc = MAX_AMPLITUDE;
	else
		sscanf (argv[1], "%d", &nc);

	if (!init_dac())
		exit(1);

	if (TOTAL_NR_OF_CHANNELS!=37)
	{
		printf("\nThis is not a 37-ch mirror\n");
		exit(0);
	}

	sprintf(mess, "\nThis program will set %d to 20-37 channels of OKO 37ch mirror.\n\n", nc);
	printf(mess);

	for (i=0; i<19; i++) voltage[i]=0;
	for (i=19; i<37; i++) voltage[i]=nc;
	set_mirror();
	close_dac();
	return 0;
}
