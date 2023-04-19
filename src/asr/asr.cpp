#include "asr.h"
#include "am.h"
#include "extract_feat.h"
#include "vad.h"
#include "stdlib.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>

#include <iostream>
#include <fstream>
#include <map>
#include <algorithm>
#include <utility>
#include <cmath>
#include "ctc_beam_search_decoder.h"
#include "scorer.h"
#include <sys/time.h>
#include <iomanip>
#include "string.h"
#include<algorithm>

#define VAD_BLOCK_SIZE (160)
#define VAD_SEGMENT_MAX (20)

#define AM_OUT_BUF_SIZE (1024*1024*10)

void convertAudioToWavBuf(
    char * toBuf, 
    char * fromBuf,
    int totalAudioLen)
{
    char * header = toBuf;
    int byteRate = 16 * 16000 * 1 / 8;
    int totalDataLen = totalAudioLen + 36;
    int channels = 1;
    int  longSampleRate = 16000;

    header[0] = 'R'; // RIFF/WAVE header
    header[1] = 'I';
    header[2] = 'F';
    header[3] = 'F';
    header[4] = (char) (totalDataLen & 0xff);
    header[5] = (char) ((totalDataLen >> 8) & 0xff);
    header[6] = (char) ((totalDataLen >> 16) & 0xff);
    header[7] = (char) ((totalDataLen >> 24) & 0xff);
    header[8] = 'W';
    header[9] = 'A';
    header[10] = 'V';
    header[11] = 'E';
    header[12] = 'f'; // 'fmt ' chunk
    header[13] = 'm';
    header[14] = 't';
    header[16] = 16; // 4 bytes: size of 'fmt ' chunk
    header[17] = 0;
    header[18] = 0;
    header[19] = 0;
    header[20] = 1; // format = 1
    header[21] = 0;
    header[22] = (char) channels;
    header[23] = 0;
    header[24] = (char) (longSampleRate & 0xff);
    header[25] = (char) ((longSampleRate >> 8) & 0xff);
    header[26] = (char) ((longSampleRate >> 16) & 0xff);
    header[27] = (char) ((longSampleRate >> 24) & 0xff);
    header[28] = (char) (byteRate & 0xff);
    header[29] = (char) ((byteRate >> 8) & 0xff);
    header[30] = (char) ((byteRate >> 16) & 0xff);
    header[31] = (char) ((byteRate >> 24) & 0xff);
    header[32] = (char) (1 * 16 / 8); // block align
    header[33] = 0;
    header[34] = 16; // bits per sample
    header[35] = 0;
    header[36] = 'd';
    header[37] = 'a';
    header[38] = 't';
    header[39] = 'a';
    header[40] = (char) (totalAudioLen & 0xff);
    header[41] = (char) ((totalAudioLen >> 8) & 0xff);
    header[42] = (char) ((totalAudioLen >> 16) & 0xff);
    header[43] = (char) ((totalAudioLen >> 24) & 0xff);

    memcpy(toBuf+44, fromBuf, totalAudioLen);

}

typedef struct
{
    float * featMean_;
    float * featStd_;
    float * amFileData_;
    void * amData_;
    void * featData_;
    void * vadData_;
    float * amOutBuf_;
    Scorer *p_score;
    int curFramePos_;
    int outDim_;
    bool vadOn_;
    bool streamOn_;
    std::vector<std::string> * vocab_;
    VAD_RESULT_t prevVadResult_;
}ASR_DATA_t;

typedef struct
{
    int len_;
    int posStart_;
    int posEnd_;
    int eos_;
    int16_t * segData_;
}SEG_INFO_t;

typedef struct
{
    int segNum_;
    SEG_INFO_t segArray[VAD_SEGMENT_MAX];
}SEGMENTS_INFO_t;

SEGMENTS_INFO_t * splitAudio(ASR_DATA_t*  asrData, int16_t * wavData, int len )
{
    SEGMENTS_INFO_t * retSegs = new SEGMENTS_INFO_t();
    memset(retSegs, 0, sizeof(SEGMENTS_INFO_t));
    
    for(int i = 0; i<VAD_SEGMENT_MAX; i++)
    {
        retSegs->segArray[i].posStart_ = 0;
        retSegs->segArray[i].posEnd_ = 0;
        retSegs->segArray[i].len_ = 0;
        retSegs->segArray[i].eos_ = 0;
        retSegs->segArray[i].segData_ = NULL;
    }

    retSegs->segArray[0].segData_ = wavData;
    retSegs->segNum_ = 0;

    int curSegIdx = 0;
    bool vadActivate = false;
    int vadBlockNum = len/(VAD_BLOCK_SIZE);
    
    for(int i = 0; i<vadBlockNum; i++)
    {
        int pos = 0;
        VAD_RESULT_t vadResult = vad_sdk_run(asrData->vadData_, (char *)(wavData+i*VAD_BLOCK_SIZE), VAD_BLOCK_SIZE*2, &pos);
                
        if(vadResult != asrData->prevVadResult_)
        {
            if(asrData->prevVadResult_ == VAD_OFF)
            {
                int startIdx = i;
                if(startIdx > 40)
                {
                    startIdx = startIdx - 40; 
                }
                else
                {
                    startIdx = 0;
                }

                retSegs->segArray[curSegIdx].posStart_ = (startIdx)*VAD_BLOCK_SIZE;
                retSegs->segArray[curSegIdx].segData_ = wavData+(startIdx)*VAD_BLOCK_SIZE; 
            }
            else 
            {
                retSegs->segArray[curSegIdx].posEnd_ = (i+1)*VAD_BLOCK_SIZE;
                retSegs->segArray[curSegIdx].len_ = retSegs->segArray[curSegIdx].posEnd_ - retSegs->segArray[curSegIdx].posStart_;
                retSegs->segArray[curSegIdx].eos_ = 1;
                if(curSegIdx < VAD_SEGMENT_MAX)
                {
                    curSegIdx = curSegIdx+1;
                }
            }
            vadActivate = true;
            asrData->prevVadResult_ = vadResult;
        }
        else if(asrData->prevVadResult_ == VAD_ON)
        {
            retSegs->segArray[curSegIdx].posEnd_ = (i+1)*VAD_BLOCK_SIZE;
            vadActivate = true;
        }
    }

    if(vadActivate == true)
    {
        retSegs->segNum_ = curSegIdx+1;

        if(retSegs->segArray[retSegs->segNum_-1].len_ == 0)
        {
            retSegs->segArray[retSegs->segNum_-1].len_ = len - retSegs->segArray[retSegs->segNum_-1].posStart_;
        }
    }
    
    return retSegs;
}

int asrLoadAMModel(char * amModelName, float ** featMean, float **featStd, float **amModel)
{
    
    struct stat st;
    if(-1 == stat(amModelName, &st))
    {
        return -1 ;
    }

    FILE *fp = fopen(amModelName, "rb");
    if(!fp)
    {
        printf("Fail to open am model file:%s!\n", amModelName);
        return -1;
    }

    float * mean = new float[161];
    float * std = new float[161];
    float * am = new float[(st.st_size/sizeof(float))-161*2];


    fread(mean,161*sizeof(float),1,fp);
    fread(std,161*sizeof(float),1,fp);
    fread(am,st.st_size-161*2*sizeof(float),1,fp);
    
    fclose(fp);

    *featMean = mean;
    *featStd = std;
    *amModel = am;
    
    return 0;
}

void * asrInit(char * amModelName, char * labelFile, char * lmFile,  bool streamOn)
{

    ASR_DATA_t * asrData = new ASR_DATA_t();
    asrLoadAMModel(amModelName, &asrData->featMean_,&asrData->featStd_,&asrData->amFileData_);

    asrData->amData_ = am_init(asrData->amFileData_,asrData->outDim_);
    asrData->featData_ = extract_feat_init(16000,10,20,asrData->featMean_, asrData->featStd_);
    asrData->vocab_ = new std::vector<std::string>(asrData->outDim_);
    
	std::ifstream v;
	v.open(labelFile);

	for (int j=0;j<asrData->outDim_;j++)
	{
	    v>>(*asrData->vocab_)[j];
	}
    v.close();

    asrData->amOutBuf_ = new float[AM_OUT_BUF_SIZE];
    memset(asrData->amOutBuf_,0,sizeof(float)*AM_OUT_BUF_SIZE);
    asrData->curFramePos_ = 0;
    asrData->vadOn_ = true;
    asrData->streamOn_ = streamOn;

    asrData->prevVadResult_ = VAD_OFF;
    if(asrData->vadOn_ == true)
    {
        asrData->vadData_ = vad_sdk_init();
    }
    else
    {
        asrData->vadData_ = NULL;
    }

	float alpha = 1.5;
    float beta = 1.0;
    asrData->p_score = new Scorer(alpha,beta,lmFile);

    return (void *)asrData;
}

std::string doDecode(ASR_DATA_t * asrData, int num_frames, float * probOut)
{
    if(asrData == NULL)
    {
        std::string strDummy="";
        return strDummy;
    }

	int num_classes = asrData->outDim_;
	int beam_size = 100;
	int blank_id = 0;
	float cutoff_prob = 0.95;
	float alpha = 1.5;
	float beta = 1.0;

	std::vector<std::vector<float> > probs_seq(num_frames,std::vector<float>(num_classes));
	std::vector<std::pair<float, std::string> > result;

	int i=0;
	while(i<num_frames)
	{
		for (int j=0;j<num_classes;j++)
		{
            probs_seq[i][j] = probOut[i*num_classes+j]; 
		}
		i++;
	}

    result=ctc_beam_search_decoder(probs_seq,beam_size,*(asrData->vocab_),blank_id,cutoff_prob,asrData->p_score);

    std::string asrResult = "";
    if(result.size() >0)
    {
	    //std::cout<<"Decoding result:"<<result[0].second<<"\n";
        asrResult = result[0].second;
    }
    
    asrResult.erase(remove(asrResult.begin(),asrResult.end(),' '),asrResult.end());
    return asrResult;
}

std::string doAsr(ASR_DATA_t * asrData, int16_t * wavData, int len, int eos, bool streamOn)
{
    float * floatBuf = new float[len];
    memset(floatBuf,0,sizeof(float)*len);

    for(int i = 0; i<len; i++)
    {
        floatBuf[i] = (float)wavData[i]/32768.0;
    }

    int frames = 0;
    float * featResult = extract_feat(asrData->featData_,floatBuf,len,&frames);

    std::string asrResult = "";
    if(featResult != NULL)
    {
        int amFrames = 0;
        float * probOut  = am_run(asrData->amData_, featResult, frames, amFrames);
        
        if((amFrames > 0)&&((asrData->outDim_*(asrData->curFramePos_+amFrames))<AM_OUT_BUF_SIZE))
        {
            memcpy(asrData->amOutBuf_+asrData->outDim_*(asrData->curFramePos_), probOut, asrData->outDim_*amFrames*sizeof(float));
            asrData->curFramePos_ = asrData->curFramePos_ + amFrames;
        }

        if((amFrames > 0)&&(streamOn == true))
        {
            asrResult = doDecode(asrData,asrData->curFramePos_, asrData->amOutBuf_);
        }

        if(eos == 1)
        {
            asrResult = doDecode(asrData,asrData->curFramePos_, asrData->amOutBuf_);
            asrData->curFramePos_ = 0;
        }

        delete []featResult;
        delete []probOut;
    }
    delete []floatBuf;

    return asrResult;
}

std::string asrRun_without_vad(void * asrdata, int16_t * wavData, int len)
{
    ASR_DATA_t * asrData = (ASR_DATA_t *)asrdata;

    if(asrData == NULL)
    {
        std::string strDummy="";
        return strDummy;
    }    

    std::string asrResult = "";
    asrResult =  doAsr(asrData, wavData, len, 1,asrData->streamOn_);
    return asrResult;
}

std::vector<std::string> asrRun_with_vad(void * asrdata, int16_t * wavData, int len)
{
    ASR_DATA_t * asrData = (ASR_DATA_t *)asrdata;

    std::vector<std::string> result;

    SEGMENTS_INFO_t * segInfo = splitAudio(asrData, wavData, len);  
    
    for(int i = 0; i< segInfo->segNum_; i++)
    {
        if(segInfo->segArray[i].posEnd_ - segInfo->segArray[i].posStart_ > 160)
        {
            std::string oneResult="";
            oneResult = doAsr(asrData, segInfo->segArray[i].segData_, segInfo->segArray[i].posEnd_ - segInfo->segArray[i].posStart_, segInfo->segArray[i].eos_,asrData->streamOn_);            
            result.push_back(oneResult);
        }
    }

    delete segInfo;
    return result;
}

void asrDestroy(void * asrdata)
{
    ASR_DATA_t * asrData = (ASR_DATA_t *)asrdata;

    if(asrData == NULL)
    {
        return;
    }    

    vad_sdk_destroy(asrData->vadData_);
    feat_extrace_destroy(asrData->featData_);
    am_destroy(asrData->amData_);     

    delete []asrData->featMean_;
    delete []asrData->featStd_;
    delete []asrData->amFileData_;
    delete []asrData->amOutBuf_;
    delete asrData->vocab_;
    delete asrData->p_score;
}
