#ifndef _EXTRACT_FEAT_H_
#define _EXTRACT_FEAT_H_

int extract_feat(float *data, int len, int fs, int strideInMs, int windowInMs, float * featOut);


#endif
