#ifndef _EXTRACT_FEAT_H_
#define _EXTRACT_FEAT_H_

void * extract_feat_init(int fs, int strideInMs, int windowInMs, float * featMean, float *featStd);
float * extract_feat(void * featDataPtr, float *data, int len, int * amFrames);
void feat_extrace_destroy(void * featDataPtr);

#endif
