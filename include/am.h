#ifndef _AM_H_
#define _AM_H_

void * am_init(float * model,int & outDim);
float * am_run(void * data, float * feat, int inputY, int & amFrames);
void am_destroy(void * data);

#endif
