#include "edac40.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define EDAC40_MAXN 10
#define EDAC40_DISCOVER_TIMEOUT 500 // milliseconds
#define EDAC40_DISCOVER_ATTEMPTS 1

/*
 ReaD the Time Stamp Counter
 This special-purpose CPU register contains
 "number of CPU cycles since reset" thus providing
 some means for high-resolutional timekeeping.
 Might work somehow different on some modern
 multicore systems and non-Intel processors.
 Asm cod is in the GNU format
*/
inline unsigned long long int rdtsc(void)
{
   unsigned a, d;
   __asm__ volatile("rdtsc" : "=a" (a), "=d" (d));
   return ((unsigned long long)a) | (((unsigned long long)d) << 32);;
}

int main()
{
   int nb;
   int tmp_ch_n;
   SOCKET edac40_socket;
   edac40_list_node edac40_list[EDAC40_MAXN];
   edac40_channel_value dac_data0[40], dac_data1[40];
   char *buf0, *buf1;
   int i,buf_len,device_num;
   unsigned long long int t1,t2;
   fprintf(stderr,"Looking for EDAC40 devices...\n");
   edac40_init();
   device_num=edac40_list_devices(edac40_list, EDAC40_MAXN, EDAC40_DISCOVER_TIMEOUT, EDAC40_DISCOVER_ATTEMPTS);
   if(device_num<=0)
        {
            fprintf(stderr,"No EDAC40 units detected.\n");
            exit(1);
        }
      else
        {
            fprintf(stderr,"Using first found EDAC40 unit for square wave generation (Ctrl-C to terminate).\n");
        }
   edac40_socket=edac40_open(edac40_list[0].IPAddress,0);
   edac40_set_timeout(edac40_socket,2000); // 2s
   tmp_ch_n=40;
   for(i=0;i<tmp_ch_n;i++)
     {
         dac_data0[i].channel=dac_data1[i].channel=i;
         dac_data0[i].value=0;
         dac_data1[i].value=0xFFFF;
     }
   edac40_prepare_packet(dac_data0,tmp_ch_n,&buf0);
   buf_len=edac40_prepare_packet(dac_data1,tmp_ch_n,&buf1);

   i=0;
   t2=rdtsc();
   while(1)
       {
        t1=rdtsc();
        if(t1-t2<800000ull) continue;
        t2=t1;
        nb=edac40_send_packet(edac40_socket, ((i++)&1)?buf0:buf1, buf_len);
        //printf("%d",nb);
        if(nb<=0)
           {
              printf("\nConnection lost.\n");
              exit(1);
           }
        //printf(".");
       }
   free(buf0);
   free(buf1);
   edac40_close(edac40_socket);
   edac40_finish();
   return 0;
}


