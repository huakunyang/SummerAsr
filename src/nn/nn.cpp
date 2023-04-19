#include "nn.h"
#include "string.h"
#include "stdio.h"
#define eps_norm (1e-05)

float * composeIM2ColBufIn(const float * Im_in,
                        int dim_im_in_x,
                        int dim_kernel_x,
                        int dim_kernel_y,
                        int ch_im_in,
                        int in_row,
                        int in_col,
                        float *Im_out)
{
    const float * pStart = Im_in + (dim_im_in_x*in_row+in_col)*ch_im_in;

    int outIdx = 0;
    for(int iy = 0; iy<dim_kernel_y; iy++)
    {
        memcpy((void *)(Im_out + outIdx),
                       (void *)(pStart+iy*dim_im_in_x*ch_im_in),
                       dim_kernel_x*ch_im_in*sizeof(float)); 

        outIdx = outIdx + dim_kernel_x*ch_im_in;                      
    }
    
    return Im_out;
}

NN_OP_STATUS_t nn_op_conv2d(float * Im_in,
                            MatrixXf & weightMatrix,
                            MatrixXf & biasMatrix,
                            float * Im_out,
                            float * tmp_buf,
                            const int dim_im_in_x,
                            const int dim_im_in_y,
                            const int ch_im_in,
                            const int ch_im_out,
                            const int dim_kernel_x,
                            const int dim_kernel_y,
                            const int padding_x,
                            const int padding_y,
                            const int stride_x,
                            const int stride_y,
                            const int dim_im_out_x,
                            float * dim_im_out_y,
                            float * remainData,
                            int * remianLines,
                            int nextLayerOffset,
                            int inited)
{

    int in_row, in_col;

    int local_py = 0;
    if(inited == 0)
    {
        local_py = padding_y;
    }

    int Ny = 0;
    int remainOffset = 0;

    if(dim_im_in_y >= dim_kernel_y)
    {
        Ny = (dim_im_in_y - dim_kernel_y + local_py)/stride_y+1;
        *remianLines = dim_im_in_y - Ny*stride_y+local_py;

        remainOffset = (dim_im_in_y - *remianLines)*dim_im_in_x*ch_im_in;
        *dim_im_out_y = (float)Ny;
        memcpy((void *)remainData,(void *)(Im_in+remainOffset),(*remianLines)*dim_im_in_x*ch_im_in*sizeof(float));
    }
    else
    {
        *remianLines = dim_im_in_y;
        *dim_im_out_y = 0;
        memcpy((void *)remainData,(void *)(Im_in+remainOffset),(*remianLines)*dim_im_in_x*ch_im_in*sizeof(float));
        return NN_OP_STATUS_SKIP;
    }

    for (int j = 0; j < (int)(*dim_im_out_y); j++)
    {
        for (int k = 0; k < dim_im_out_x; k++)
        {
            int in_row = stride_y * j  - local_py;
            int in_col = stride_x * k  - padding_x;

            float * im2bufIn = composeIM2ColBufIn(Im_in,
                                                        dim_im_in_x, 
                                                        dim_kernel_x, 
                                                        dim_kernel_y, 
                                                        ch_im_in, 
                                                        in_row, 
                                                        in_col,
                                                        tmp_buf); 

            MatrixXf imInMat =  Map<MatrixXf>(im2bufIn,1,ch_im_in*dim_kernel_x*dim_kernel_y);
            MatrixXf chOutMat = imInMat * weightMatrix + biasMatrix;
            chOutMat = ((((chOutMat.array().pow(3)*0.044715+chOutMat.array())*0.7978845608028654).tanh()+1)*chOutMat.array()*0.5).matrix();

            memcpy(Im_out+nextLayerOffset+(j * dim_im_out_x + k) * ch_im_out,chOutMat.data(),ch_im_out*sizeof(float));
        }
    }

    return NN_OP_STATUS_OK;
}

MatrixXf sigmod(MatrixXf val)
{
    MatrixXf oneMat = MatrixXf::Ones(val.rows(),val.cols());
    return (oneMat.array()/((val.array()*(-1.0)).exp()+1.0)).matrix();
}

NN_OP_STATUS_t nn_op_gru(float * im_in,
                         float * prevHiddenState,
                         int input_size,
                         int hidden_size,
                         int seq_length,
                         MatrixXf & x2hWeightMatrix,
                         MatrixXf & h2hWeightMatrix,
                         MatrixXf & x2hBias,
                         MatrixXf & h2hBias,
                         float * curHiddenState,
                         float * out_state)
{

    float * hidden = (float *)malloc(sizeof(float) * hidden_size);
    memcpy(hidden,prevHiddenState,sizeof(float) * hidden_size); 

    for(int seqIdx = 0; seqIdx < seq_length; seqIdx++)
    {
        MatrixXf inputXMatrix = Map<MatrixXf>(im_in + seqIdx*input_size,input_size,1);
        MatrixXf gate_x = x2hWeightMatrix * inputXMatrix + x2hBias;

        MatrixXf hiddenMatrix = Map<MatrixXf>(hidden,hidden_size,1);
        MatrixXf gate_h = h2hWeightMatrix * hiddenMatrix + h2hBias;

        MatrixXf i_r = gate_x.block(0,0,hidden_size,1);
        MatrixXf i_i = gate_x.block(hidden_size,0,hidden_size,1);
        MatrixXf i_n = gate_x.block(hidden_size*2,0,hidden_size,1);
        
        MatrixXf h_r = gate_h.block(0,0,hidden_size,1);
        MatrixXf h_i = gate_h.block(hidden_size,0,hidden_size,1);
        MatrixXf h_n = gate_h.block(hidden_size*2,0,hidden_size,1);
        
        MatrixXf resetgate = sigmod(i_r + h_r);
        MatrixXf inputgate = sigmod(i_i + h_i);

        MatrixXf newgate = ((resetgate.array()*h_n.array() + i_n.array()).tanh()).matrix();

        MatrixXf hy = (newgate.array() + inputgate.array() * (hiddenMatrix.array() - newgate.array())).matrix();

        memcpy(out_state + seqIdx*hidden_size,hy.data(),sizeof(float)*hidden_size);
        memcpy(hidden,hy.data(),sizeof(float)*hidden_size);
    }

    memcpy(curHiddenState, hidden, sizeof(float)*hidden_size);
    free(hidden);

    return NN_OP_STATUS_OK;
}                         

NN_OP_STATUS_t nn_op_norm(float * im_in,
                          MatrixXf & normWMat,
                          MatrixXf & normBiasMat,
                          int len,
                          int seq_len,
                          float * out)
{
    for(int i  = 0; i < seq_len; i++)
    {
        MatrixXf m = Map<MatrixXf>(im_in + i*len, 1, len);
        Eigen::MatrixXf mean = m.rowwise().mean();
        float mean_ = mean(0, 0);
        Eigen::MatrixXf sqsum = (m * m.transpose()).rowwise().sum();
        float sqsum_ = sqsum(0, 0);
        float scale = 1. /(float)len;
        float variance_ = sqsum_ * scale - mean_ * mean_;

        MatrixXf lineNormed =  (((m.array() - mean_)/sqrt(variance_ + eps_norm))*normWMat.array()+normBiasMat.array()).matrix();
        memcpy(out+i*len,lineNormed.data(),sizeof(float)*len);
    }
    return NN_OP_STATUS_OK;    
}

NN_OP_STATUS_t nn_op_fc(float * im_in,
                        MatrixXf & weightMat,
                        MatrixXf & biasMat,
                        int wRow,
                        int wCol,
                        int inRow,
                        int inCol,
                        float * out)
{
    MatrixXf inMat = MatrixXf(inRow,inCol);

    for(int i = 0; i<inRow; i++)
    {
        MatrixXf row = Map<MatrixXf>(im_in + i*inCol, 1, inCol);
        inMat.block(i,0,1,inCol) = row.block(0,0,1,inCol);
    }

    MatrixXf outMat = inMat*(weightMat); 
    
    for(int i = 0; i< inRow; i++)
    {
       outMat.row(i) = outMat.row(i) + biasMat;
    }
    
    MatrixXf outMatT = outMat.transpose();
    
    memcpy(out, outMatT.data(), outMatT.rows()*outMatT.cols()*sizeof(float));    
    return NN_OP_STATUS_OK;    
}

NN_OP_STATUS_t nn_op_softmax(float * im_in,
                             int row,
                             int col,
                             float * out)
{
    for(int i = 0; i< row; i++)
    {
        MatrixXf row = Map<MatrixXf>(im_in + i*col,1,col);
        MatrixXf m = row.array().exp();
        float sum = m.sum();
        MatrixXf n = m / sum;
        memcpy(out+i*col,n.data(),col*sizeof(float));
    }
    return NN_OP_STATUS_OK;
}
