#ifndef _ASR_H_
#define _ASR_H_

#include "stdint.h"
#include <string>
#include <vector>

using namespace std;

void * asrInit(char * amModelName, char * labelFile, char * lmFile,  bool streamOn);
std::vector<std::string> asrRun_with_vad(void * asrdata, int16_t * wavData, int len);
std::string  asrRun_without_vad(void * asrdata, int16_t * wavData, int len);
void asrDestroy(void * asrdata);

#endif
