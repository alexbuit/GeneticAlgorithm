#include <stdio.h>
#include "mirror_edac40.h"

extern int addr[TOTAL_NR_OF_CHANNELS];
extern int unit[TOTAL_NR_OF_CHANNELS];

int main()
{
    print_pinout();
    return 0;
}
