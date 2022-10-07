#include <stdio.h>
#include "edac40.h"
#define EDAC40_MAXN 10
#define EDAC40_DISCOVER_TIMEOUT 1500 // milliseconds
#define EDAC40_DISCOVER_ATTEMPTS 1

int main()
{
   int device_num,i;
   edac40_list_node edac40_list[EDAC40_MAXN];
   edac40_init();
   printf("Detecting EDAC40 devices...\n");
   if((device_num=edac40_list_devices(edac40_list, EDAC40_MAXN, EDAC40_DISCOVER_TIMEOUT, EDAC40_DISCOVER_ATTEMPTS))>=0) printf("Detected %d EDAC40 unit(s).\n",device_num);
	  else printf("Error detecting devices\n");
   for(i=0; i<device_num; i++)
      printf("Unit %d: IP Address: %s, MAC Address:%s\n",i,edac40_list[i].IPAddress,edac40_list[i].MACAddress);
   edac40_finish();
   return 0;
}

