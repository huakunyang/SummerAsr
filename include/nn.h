#ifndef _NN_H_
#define _NN_H_
#include <Eigen/Dense>

using Eigen::MatrixXf;
using Eigen::MatrixXd;
using Eigen::Map;
using Eigen::ArrayXXf;
using Eigen::Array;
using Eigen::Dynamic;
using namespace std;

typedef enum
{
    NN_OP_STATUS_SKIP,
    NN_OP_STATUS_OK  

}NN_OP_STATUS_t;


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
                            int inited);

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
                         float * out_state);

NN_OP_STATUS_t nn_op_norm(float * im_in,
                          MatrixXf & normWMat,
                          MatrixXf & normBiasMat,
                          int len,
                          int seq_len,
                          float * out);

NN_OP_STATUS_t nn_op_fc(float * im_in,
                        MatrixXf & weightMat,
                        MatrixXf & biasMat,
                        int wRow,
                        int wCol,
                        int inRow,
                        int inCol,
                        float * out);

NN_OP_STATUS_t nn_op_softmax(float * im_in,
                             int row,
                             int col,
                             float * out);
#endif
