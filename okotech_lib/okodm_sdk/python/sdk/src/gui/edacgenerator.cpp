#include "edacgenerator.h"
#include "edac40.h"
//#include <winsock.h>
#include <stdint.h>

EDACGenerator::EDACGenerator()
{
  stopped=false;
}

void EDACGenerator::stop()
{
    stopped=true;
}

void EDACGenerator::run()
{
  char *e40buf0, *e40buf1;
  unsigned buf_len,i;
  edac40_channel_value e40data0[40],e40data1[40];
  unsigned long long t1,t2;
  for(i=0;i<40;i++)
    {
      e40data0[i].channel=e40data1[i].channel=i;
      e40data0[i].value=waveAmplitude;
      e40data1[i].value=0;
    }
  edac40_prepare_packet(e40data0,40,&e40buf0);
  buf_len=edac40_prepare_packet(e40data1,40,&e40buf1);
  if(waveMode==0) memset2(e40buf1+6,waveAmplitude,(buf_len-6)/2);
  i=0;
  t2=rdtsc();
  if(waveMode==1 || waveMode==0) // square wave or constant
    {
      while(!stopped)
        {
          t1=rdtsc();
          if(t1-t2<800000ull) continue;
          t2=t1;
          edac40_send_packet(e40socket, ((i++)&1)?e40buf0:e40buf1, buf_len);
        }
    }
  if(waveMode==2) // linear ramp
    {
      while(!stopped)
        {
          t1=rdtsc();
          if(t1-t2<800000ull) continue;
          memset2(e40buf0+6,i,(buf_len-6)/2);
          edac40_send_packet(e40socket, e40buf0, buf_len);
          t2=t1;
          if((i+=100)>=waveAmplitude) i=0;
        }
    }
  stopped=false;
  free(e40buf0);
  free(e40buf1);
}

void EDACGenerator::setParameters(SOCKET socket, int mode, unsigned int amplitude)
{
  e40socket=socket;
  waveMode=mode;
  waveAmplitude=amplitude;
}

// ReaD the Time Stamp Counter
inline unsigned long long int rdtsc(void)
{
   unsigned a, d;
   __asm__ volatile("rdtsc" : "=a" (a), "=d" (d));
   return ((unsigned long long)a) | (((unsigned long long)d) << 32);;
}

