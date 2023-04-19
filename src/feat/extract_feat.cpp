#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>
#include <vector>
#include <complex>
#include <stdlib.h>
#include <math.h>
#include <iostream>

#define eps (1e-14)
#define eps_featnorm (1e-20)

extern float hanning[];

using Eigen::MatrixXf;
using Eigen::MatrixXd;
using Eigen::Map;
using Eigen::ArrayXXf;
using Eigen::Array;
using Eigen::Dynamic;
using namespace std;

typedef struct
{
    float * featMean_;
    float * featStd_;
    float * prevStride_;
    float * tmpBuf_;
    float * tmpBuf2_;
    int fs_;
    int strideInMs_;
    int windowInMs_;
    int init_;

}FEAT_EXTRACT_DATA_t;

float rms_db(float *data, int len)
{
    float powerSum = 0;
    MatrixXf power = Map<MatrixXf>(data,1,len);
    return 10*log10(power.array().square().sum()/(float)len);
}

void normalize(float *data, int len, float rmsDb, float targetDb)
{
    MatrixXf inMatrix = Map<MatrixXf>(data,1,len);
    float gain = targetDb - rmsDb;
    
    float adjustVal = pow(10,(gain/(float)20));
    inMatrix = inMatrix * adjustVal;
    memcpy(data,inMatrix.data(),sizeof(float)*len);
}

void * extract_feat_init(int fs, int strideInMs, int windowInMs, float * featMean, float *featStd)
{
    FEAT_EXTRACT_DATA_t * featData = new FEAT_EXTRACT_DATA_t();
    memset(featData,0,sizeof(FEAT_EXTRACT_DATA_t));

    featData->fs_ = fs;
    featData->strideInMs_ = strideInMs;
    featData->windowInMs_ = windowInMs;
    featData->featMean_ = featMean;
    featData->featStd_ = featStd;    

    int strideLen = (int)(0.001 * fs * strideInMs);
    int windowLen = (int)(0.001 * featData->fs_ * featData->windowInMs_);
    featData->prevStride_ = new float[strideLen*2];
    memset(featData->prevStride_,0,sizeof(float)*strideLen*2);

    featData->tmpBuf_ = new float[windowLen];
    memset(featData->tmpBuf_, 0, sizeof(float)*windowLen);

    featData->tmpBuf2_ = new float[windowLen/2+1];
    memset(featData->tmpBuf2_,0,sizeof(float)*(windowLen/2+1));

    return (void *) featData;
}

float * extract_feat(void * featDataPtr, float *data, int len, int *amFrames)
{
    FEAT_EXTRACT_DATA_t * featData = (FEAT_EXTRACT_DATA_t *)featDataPtr;

    if(featData == NULL)
    {
        * amFrames = 0;
        return NULL;
    }

    int strideLen = (int)(0.001 * featData->fs_ * featData->strideInMs_);
    int windowLen = (int)(0.001 * featData->fs_ * featData->windowInMs_);
    int truncateLen = (len - windowLen) % strideLen;
    int frames = (int)((len - truncateLen)/strideLen);

    Eigen::FFT<float> fft;
    MatrixXf hanningData = Map<MatrixXf>(hanning,1,windowLen);

    float scale = hanningData.array().square().sum()*featData->fs_;
    float scale2 = 2.0/scale;

    float rmsDB = rms_db(data,len);
    normalize(data,len,rmsDB,-20.0);        

    MatrixXf featMeanMatrix = Map<MatrixXf>(featData->featMean_,1,161);
    MatrixXf featStdMatrix =  Map<MatrixXf>(featData->featStd_,1,161);

    int outIdx = 0;
    int retFrames = frames;
    if(featData->init_ == 0)
    {
        retFrames = retFrames -1 ;
    }

    float * featOut = new float[frames*(windowLen/2+1)];
    memset(featOut, 0 , sizeof(float)*frames*(windowLen/2+1));

    for(int i = 0; i< frames; i++)
    {
        if(featData->init_ == 0)
        {
            memcpy(featData->prevStride_,data,2*strideLen*sizeof(float));
            i = i +1;
            featData->init_ = 1;
        }
        else
        {
            memcpy(featData->prevStride_+strideLen,data+i*strideLen,strideLen*sizeof(float));
        }
        MatrixXf frameData = Map<MatrixXf>(featData->prevStride_, 1, windowLen);
        MatrixXf frameHanmming = (frameData.array() * hanningData.array()).matrix();

        std::vector<float> vecHanmming;
        vecHanmming.resize(frameHanmming.cols());

        for(int kk = 0; kk<frameHanmming.cols(); kk++)
        {
            vecHanmming[kk] = frameHanmming(0,kk);
        }        

        std::vector<std::complex<float> > freqvec;
        fft.fwd(freqvec,vecHanmming);

        memset(featData->tmpBuf_,0,sizeof(float)*windowLen);

        for(int j = 0; j< windowLen; j++)
        {
            featData->tmpBuf_[j] = std::abs(freqvec[j]);            
        }

        MatrixXf fftMat = Map<MatrixXf>(featData->tmpBuf_, 1, windowLen);
        fftMat = fftMat.array().square().matrix();
        
        MatrixXf midFftMat = fftMat.block(0,1,1,windowLen/2-1);

        midFftMat = (midFftMat.array()*scale2+eps).log().matrix();

        memcpy(featData->tmpBuf2_+1,midFftMat.data(),sizeof(float)*(windowLen/2));
        featData->tmpBuf2_[0] = log(fftMat.data()[0]/scale+eps);
        featData->tmpBuf2_[windowLen/2] = log(fftMat.data()[windowLen/2]/scale+eps);

        MatrixXf featTmp = Map<MatrixXf>(featData->tmpBuf2_,1,161);

        MatrixXf featOutMat = ((featTmp.array() - featMeanMatrix.array())/(featStdMatrix.array()+eps_featnorm)).matrix();
        memcpy(featOut+(outIdx++)*(windowLen/2+1),featOutMat.data(),sizeof(float)*(windowLen/2+1));

        memcpy(featData->prevStride_,featData->prevStride_+strideLen,strideLen*sizeof(float));
    }
    *amFrames = retFrames;

    return featOut;
}

void feat_extrace_destroy(void * featDataPtr)
{
    FEAT_EXTRACT_DATA_t * featData = (FEAT_EXTRACT_DATA_t *)featDataPtr;

    if(featData == NULL)
    {
        return;
    }
    
    delete []featData->prevStride_;
    delete []featData->tmpBuf_;
    delete []featData->tmpBuf2_;
    delete featData;
}
