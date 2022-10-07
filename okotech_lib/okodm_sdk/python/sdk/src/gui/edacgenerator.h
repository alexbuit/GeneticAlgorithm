#ifndef EDACGENERATOR_H
#define EDACGENERATOR_H

#include "edac40.h"
#include <stdint.h>
#include <QThread>

class EDACGenerator : public QThread
{
public:
  EDACGenerator();
  void setParameters(SOCKET socket, int mode, unsigned int amplitude);
  void stop();
protected:
  void run();
private:
  volatile bool stopped;
  SOCKET e40socket;
  unsigned int waveAmplitude;
  int waveMode;
};

unsigned long long int rdtsc(void);

#endif // EDACGENERATOR_H
