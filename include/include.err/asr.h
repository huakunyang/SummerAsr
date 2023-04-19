#ifndef _ASR_H_
#define _ASR_H_

void * asrInit(float * amModel);
void asrRun(void * asrdata, float * wavData, int len);

#endif
