#ifndef _AM_H_
#define _AM_H_

void * am_init(float * model);
int am_run(void * data, float * feat, int inputY, float * out);

#endif
