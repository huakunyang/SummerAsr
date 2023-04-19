#include "am.h"
#include "stdlib.h"
#include "string.h"
#include "nn.h"

#define TMP_BUF_SIZE (1024*1024*20)
#define BUF1_SIZE (1024*1024*20)
#define BUF2_SIZE (1024*1024*20)

typedef struct
{
    float inX_;
    float inY_;
    float inCh_;
    float outCh_;
    float outX_;
    float outY_;
    float kX_;
    float kY_;
    float sX_;
    float sY_;
    float pX_;
    float pY_;
    float relu_;
    float biasShift_;
    float outShift_;
}NN_CONV_INFO_t;

typedef struct
{
    NN_CONV_INFO_t nnInfo_;
    float * convWt_;
    float * bias_;
    float * remainData_;
    int init_;
    int remainLines_;
    MatrixXf weightMatrix_;
    MatrixXf biasMatrix_;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
}NN_CONV_PARAMS_t;

typedef struct
{
    float hiddenSize_;
    float inputSize_;
}NN_RNN_INFO_t;

typedef struct
{
    NN_RNN_INFO_t nnInfo_;
    float * weightIh_;
    float * weightHh_;
    float * biasIh_;
    float * biasHh_;
    float * normWeight_;
    float * normBias_;
    float * prevHiddenStates_;
    MatrixXf x2hWeightMatrix_;
    MatrixXf h2hWeightMatrix_;
    MatrixXf x2hBias_;
    MatrixXf h2hBias_;
    MatrixXf normWMat_;
    MatrixXf normBiasMat_;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
}NN_RNN_PARAMS_t;

typedef struct
{
    float outDim_;
    float inDim_;
}NN_FC_INFO_t; 

typedef struct
{
    NN_FC_INFO_t nnInfo_;
    float * weight_;
    float * bias_;
    MatrixXf weightMat_;
    MatrixXf biasMat_;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
}NN_FC_PARAMS_t;

typedef struct
{
    float convNum_;
    float rnnNum_;
    float fcNum_;
    float outDim_;
}AM_HEADER_INFO_t;

typedef struct
{
    AM_HEADER_INFO_t amHeader_;
    NN_CONV_PARAMS_t ** convLayers_;
    NN_RNN_PARAMS_t  ** rnnLayers_;
    NN_FC_PARAMS_t ** fcLayers_;
    float tmpBuf_[TMP_BUF_SIZE];    
    float buffer1_[BUF1_SIZE];
    float buffer2_[BUF2_SIZE];
}AM_NN_DATA_t;

void * am_init(float * model,int & outDim)
{
    AM_NN_DATA_t * amData = new AM_NN_DATA_t();
    memset(amData,0,sizeof(AM_NN_DATA_t));

    memcpy(&(amData->amHeader_),model,sizeof(AM_HEADER_INFO_t));
    outDim = (int)amData->amHeader_.outDim_;
   
    amData->convLayers_ = (NN_CONV_PARAMS_t **)malloc(sizeof(NN_CONV_PARAMS_t *)*(int)(amData->amHeader_.convNum_));
    amData->rnnLayers_ = (NN_RNN_PARAMS_t **)malloc(sizeof(NN_RNN_PARAMS_t *)*(int)(amData->amHeader_.rnnNum_));
    amData->fcLayers_ = (NN_FC_PARAMS_t **)malloc(sizeof(NN_FC_PARAMS_t *)*(int)(amData->amHeader_.fcNum_));

    int index = sizeof(AM_HEADER_INFO_t)/sizeof(float);
    for(int i = 0; i< (int)(amData->amHeader_.convNum_); i++)
    {
        amData->convLayers_[i] = new NN_CONV_PARAMS_t();

        memset(amData->convLayers_[i],0,sizeof(NN_CONV_PARAMS_t));
        memcpy(&(amData->convLayers_[i]->nnInfo_),model+index,sizeof(NN_CONV_INFO_t));
        index = index + sizeof(NN_CONV_INFO_t)/sizeof(float);
        
        amData->convLayers_[i]->convWt_ = model+index;

        amData->convLayers_[i]->weightMatrix_ = Map<MatrixXf>(amData->convLayers_[i]->convWt_,
                                                              (int)(amData->convLayers_[i]->nnInfo_.inCh_*
                                                              amData->convLayers_[i]->nnInfo_.kX_*
                                                              amData->convLayers_[i]->nnInfo_.kY_),
                                                              (int)(amData->convLayers_[i]->nnInfo_.outCh_));

        index = index + (int)(amData->convLayers_[i]->nnInfo_.inCh_  * 
                        amData->convLayers_[i]->nnInfo_.outCh_ *
                        amData->convLayers_[i]->nnInfo_.kX_    *
                        amData->convLayers_[i]->nnInfo_.kY_) ;

        amData->convLayers_[i]->bias_ = model+index;
        amData->convLayers_[i]->biasMatrix_ = Map<MatrixXf>(amData->convLayers_[i]->bias_,
                                                            1,
                                                            (int)(amData->convLayers_[i]->nnInfo_.outCh_));

        index = index + (int)amData->convLayers_[i]->nnInfo_.outCh_;

        amData->convLayers_[i]->remainData_ = new float[(int)(amData->convLayers_[i]->nnInfo_.kY_ *
                                                     amData->convLayers_[i]->nnInfo_.inCh_*
                                                     amData->convLayers_[i]->nnInfo_.inX_)];
        
        memset(amData->convLayers_[i]->remainData_, 
              0 , 
              sizeof(float) *
              amData->convLayers_[i]->nnInfo_.kY_ *
              amData->convLayers_[i]->nnInfo_.inCh_*
              amData->convLayers_[i]->nnInfo_.inX_);

    }

    for(int i = 0; i< (int)(amData->amHeader_.rnnNum_); i++)
    {
        amData->rnnLayers_[i] = new NN_RNN_PARAMS_t();

        memset(amData->rnnLayers_[i],0,sizeof(NN_RNN_PARAMS_t));
        memcpy(&(amData->rnnLayers_[i]->nnInfo_),model+index,sizeof(NN_RNN_INFO_t));
        index = index + sizeof(NN_RNN_INFO_t)/sizeof(float);
        
        amData->rnnLayers_[i]->weightIh_ = model+index;

        amData->rnnLayers_[i]->x2hWeightMatrix_ = Map<MatrixXf>(amData->rnnLayers_[i]->weightIh_,
                                                      (int)(amData->rnnLayers_[i]->nnInfo_.hiddenSize_*3),
                                                      (int)(amData->rnnLayers_[i]->nnInfo_.inputSize_));

        index = index + (int)(amData->rnnLayers_[i]->nnInfo_.inputSize_ *
                             3 *
                             amData->rnnLayers_[i]->nnInfo_.hiddenSize_);
        
        amData->rnnLayers_[i]->weightHh_ = model+index;

        amData->rnnLayers_[i]->h2hWeightMatrix_ = Map<MatrixXf>(amData->rnnLayers_[i]->weightHh_,
                                                                (int)(amData->rnnLayers_[i]->nnInfo_.hiddenSize_*3),
                                                                (int)(amData->rnnLayers_[i]->nnInfo_.hiddenSize_));

        index = index +(int)(amData->rnnLayers_[i]->nnInfo_.hiddenSize_ *
                             3 *
                             amData->rnnLayers_[i]->nnInfo_.hiddenSize_);

        amData->rnnLayers_[i]->biasIh_ = model+index;

        amData->rnnLayers_[i]->x2hBias_ = Map<MatrixXf>(amData->rnnLayers_[i]->biasIh_,
                                                        (int)(amData->rnnLayers_[i]->nnInfo_.hiddenSize_*3), 1);

        index = index + (int)(amData->rnnLayers_[i]->nnInfo_.hiddenSize_*3);

        amData->rnnLayers_[i]->biasHh_ = model+index;

        amData->rnnLayers_[i]->h2hBias_ = Map<MatrixXf>(amData->rnnLayers_[i]->biasHh_,
                                                        (int)(amData->rnnLayers_[i]->nnInfo_.hiddenSize_*3),1);

        index = index + (int)(amData->rnnLayers_[i]->nnInfo_.hiddenSize_*3);

        amData->rnnLayers_[i]->normWeight_ = model+index;

        amData->rnnLayers_[i]->normWMat_ = Map<MatrixXf>(amData->rnnLayers_[i]->normWeight_,
                                                         1,
                                                         (int)amData->rnnLayers_[i]->nnInfo_.hiddenSize_);

        index = index + (int)amData->rnnLayers_[i]->nnInfo_.hiddenSize_;
        
        amData->rnnLayers_[i]->normBias_ = model+index;
        amData->rnnLayers_[i]->normBiasMat_ = Map<MatrixXf>(amData->rnnLayers_[i]->normBias_,
                                                            1,
                                                            (int)amData->rnnLayers_[i]->nnInfo_.hiddenSize_);

        index = index +(int)amData->rnnLayers_[i]->nnInfo_.hiddenSize_;

        amData->rnnLayers_[i]->prevHiddenStates_ = new float[(int)amData->rnnLayers_[i]->nnInfo_.hiddenSize_];
        memset(amData->rnnLayers_[i]->prevHiddenStates_,0,(int)amData->rnnLayers_[i]->nnInfo_.hiddenSize_*sizeof(float));
    } 

    for(int i = 0; i< (int)amData->amHeader_.fcNum_; i++)
    {
        amData->fcLayers_[i] = new NN_FC_PARAMS_t();
        memset(amData->fcLayers_[i],0,sizeof(NN_FC_PARAMS_t));
        memcpy(&(amData->fcLayers_[i]->nnInfo_),model+index,sizeof(NN_FC_INFO_t));
        index  = index + sizeof(NN_FC_INFO_t)/sizeof(float);

        amData->fcLayers_[i]->weight_ = model + index;
        amData->fcLayers_[i]->weightMat_ = Map<MatrixXf>(amData->fcLayers_[i]->weight_,
                                                         (int)amData->fcLayers_[i]->nnInfo_.inDim_,
                                                         (int)amData->fcLayers_[i]->nnInfo_.outDim_);


        index = index + (int)(amData->fcLayers_[i]->nnInfo_.inDim_ *
                              amData->fcLayers_[i]->nnInfo_.outDim_);

        amData->fcLayers_[i]->bias_ = model + index;
        amData->fcLayers_[i]->biasMat_ = Map<MatrixXf>(amData->fcLayers_[i]->bias_,1,(int)amData->fcLayers_[i]->nnInfo_.outDim_);

        index = index + (int)(amData->fcLayers_[i]->nnInfo_.outDim_);
    
    }
    return (void *)amData; 
}

void switchBuf(float ** nn_input_data, float** nn_output_data, float * buffer1, float * buffer2)
{
    *nn_input_data = *nn_output_data;
    if(*nn_input_data == buffer1)
    {
        *nn_output_data = buffer2;
    }
    else
    {
        *nn_output_data = buffer1;
    }
}

float * am_run(void * data, float * feat, int inputY, int & amFrames)
{
    AM_NN_DATA_t * amData = (AM_NN_DATA_t *)data;

    if(amData == NULL)
    {
        amFrames = -1;
        return NULL;
    }

    memcpy(amData->buffer1_, 
           amData->convLayers_[0]->remainData_, 
           (int)amData->convLayers_[0]->remainLines_* (int)amData->convLayers_[0]->nnInfo_.inX_ * (int)amData->convLayers_[0]->nnInfo_.inCh_*sizeof(float));

    float * inputBuf = (float *)(amData->buffer1_+(amData->convLayers_[0]->remainLines_*(int)amData->convLayers_[0]->nnInfo_.inX_*(int)amData->convLayers_[0]->nnInfo_.inCh_));

    memcpy(inputBuf,feat,amData->convLayers_[0]->nnInfo_.inX_*inputY*(int)amData->convLayers_[0]->nnInfo_.inCh_*sizeof(float));
    
    float * nn_input_data =  amData->buffer1_;
    float * nn_output_data = amData->buffer2_;

    int nextLayerOffset = 0;

    int binConvInputY = inputY;

    for(int i = 0; i<(int)amData->amHeader_.convNum_;i++)
    {
        int firstConv = amData->convLayers_[i]->init_;

        if(amData->convLayers_[i]->init_ == 0)
        {
            amData->convLayers_[i]->init_ = 1;
        }

        if(i < (int)amData->amHeader_.convNum_-1)
        {
            nextLayerOffset = amData->convLayers_[i+1]->remainLines_* 
            (int)amData->convLayers_[i+1]->nnInfo_.inX_* 
            (int)amData->convLayers_[i+1]->nnInfo_.inCh_;
        }
        else
        {
            nextLayerOffset = 0;
        }

        if(NN_OP_STATUS_SKIP == nn_op_conv2d(nn_input_data,
                                             amData->convLayers_[i]->weightMatrix_,
                                             amData->convLayers_[i]->biasMatrix_,
                                             nn_output_data,
                                             amData->tmpBuf_,
                                             (int)amData->convLayers_[i]->nnInfo_.inX_,
                                             (int)(binConvInputY + amData->convLayers_[i]->remainLines_),
                                             (int)amData->convLayers_[i]->nnInfo_.inCh_,
                                             (int)amData->convLayers_[i]->nnInfo_.outCh_,
                                             (int)amData->convLayers_[i]->nnInfo_.kX_,  
                                             (int)amData->convLayers_[i]->nnInfo_.kY_,
                                             (int)amData->convLayers_[i]->nnInfo_.pX_,
                                             (int)amData->convLayers_[i]->nnInfo_.pY_,
                                             (int)amData->convLayers_[i]->nnInfo_.sX_,
                                             (int)amData->convLayers_[i]->nnInfo_.sY_,
                                             (int)amData->convLayers_[i]->nnInfo_.outX_,
                                             &(amData->convLayers_[i]->nnInfo_.outY_),
                                             amData->convLayers_[i]->remainData_,
                                             &(amData->convLayers_[i]->remainLines_),
                                             nextLayerOffset,
                                             firstConv))
        {
            amData->convLayers_[i]->init_ = firstConv;
            amFrames =-1;
            return NULL;
        }
    
        if(amData->convLayers_[i]->nnInfo_.outY_ < 1)
        {
            amFrames =-1;
            return NULL;
        }

        binConvInputY = (int)amData->convLayers_[i]->nnInfo_.outY_;

        if(i < (amData->amHeader_.convNum_ -1))
        {
            memcpy(nn_output_data,amData->convLayers_[i+1]->remainData_, amData->convLayers_[i+1]->remainLines_*(int)amData->convLayers_[i+1]->nnInfo_.inX_*(int)amData->convLayers_[i+1]->nnInfo_.inCh_*sizeof(float));
            
        }
        
        switchBuf(&nn_input_data, &nn_output_data,amData->buffer1_,amData->buffer2_);
    }

    
    int convOutX = (int)amData->convLayers_[(int)amData->amHeader_.convNum_-1]->nnInfo_.outX_;
    int convOutY = (int)amData->convLayers_[(int)amData->amHeader_.convNum_-1]->nnInfo_.outY_;
    int convOutCh = (int)amData->convLayers_[(int)amData->amHeader_.convNum_-1]->nnInfo_.outCh_;

    for(int i = 0; i<convOutY; i++)
    {
        for(int j = 0; j< convOutCh; j++)
        {
            for(int k = 0; k<convOutX; k++)
            {
                nn_output_data[i*(convOutCh*convOutX) + j*convOutX + k] = nn_input_data[i*(convOutCh*convOutX) + k*convOutCh + j];
            }
        }
    }

    switchBuf(&nn_input_data, &nn_output_data,amData->buffer1_,amData->buffer2_);

    for(int i = 0; i< (int)amData->amHeader_.rnnNum_; i++)
    {
        nn_op_gru(nn_input_data,
                  amData->rnnLayers_[i]->prevHiddenStates_,
                     amData->rnnLayers_[i]->nnInfo_.inputSize_,
                     amData->rnnLayers_[i]->nnInfo_.hiddenSize_,
                     convOutY,
                     amData->rnnLayers_[i]->x2hWeightMatrix_,
                     amData->rnnLayers_[i]->h2hWeightMatrix_,
                     amData->rnnLayers_[i]->x2hBias_,
                     amData->rnnLayers_[i]->h2hBias_,
                     amData->rnnLayers_[i]->prevHiddenStates_,
                     nn_output_data);

        switchBuf(&nn_input_data, &nn_output_data,amData->buffer1_,amData->buffer2_);

        nn_op_norm(nn_input_data,
                   amData->rnnLayers_[i]->normWMat_,
                   amData->rnnLayers_[i]->normBiasMat_,
                   amData->rnnLayers_[i]->nnInfo_.hiddenSize_,
                   convOutY,
                   nn_output_data);

        switchBuf(&nn_input_data, &nn_output_data,amData->buffer1_,amData->buffer2_);
    }

    for(int i =0; i< amData->amHeader_.fcNum_; i++)
    {
        nn_op_fc(nn_input_data,
                 amData->fcLayers_[i]->weightMat_,
                 amData->fcLayers_[i]->biasMat_,
                 (int)amData->amHeader_.outDim_,
                 (int)amData->rnnLayers_[(int)amData->amHeader_.rnnNum_-1]->nnInfo_.hiddenSize_,
                 convOutY,
                 (int)amData->rnnLayers_[(int)amData->amHeader_.rnnNum_-1]->nnInfo_.hiddenSize_,
                 nn_output_data);

        switchBuf(&nn_input_data, &nn_output_data,amData->buffer1_,amData->buffer2_);
    }    

    float * out = new float[convOutY*(int)amData->amHeader_.outDim_];
    memset(out,0,sizeof(float)*convOutY*(int)amData->amHeader_.outDim_);
    
    nn_op_softmax(nn_input_data, 
                  convOutY,
                  (int)amData->amHeader_.outDim_,
                  out);

    amFrames = convOutY;
    
    return out;
}

void am_destroy(void * data)
{
    AM_NN_DATA_t * amData = (AM_NN_DATA_t *)data;

    if(amData == NULL)
    {
        return;
    }
    
    for(int i = 0; i< (int)(amData->amHeader_.convNum_); i++)
    {
        delete amData->convLayers_[i]; 
    }

    
    for(int i = 0; i< (int)(amData->amHeader_.rnnNum_); i++)
    {
        delete []amData->rnnLayers_[i]->prevHiddenStates_;
        delete amData->rnnLayers_[i];  
    }

    for(int i = 0; i< (int)amData->amHeader_.fcNum_; i++)
    {
        delete amData->fcLayers_[i];
    }

    free(amData->convLayers_);
    free(amData->rnnLayers_);
    free(amData->fcLayers_);
    delete amData;
}
