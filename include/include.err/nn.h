#ifndef _NN_H_
#define _NN_H_

typedef enum
{
    NN_OP_STATUS_SKIP,
    NN_OP_STATUS_OK  

}NN_OP_STATUS_t;

NN_OP_STATUS_t nn_op_conv2d(float * Im_in,
                            float * wt,
                            float * bias,
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
                         int input_size,
                         int hidden_size,
                         int seq_length,
                         float * weight_x2h,
                         float * weight_h2h,
                         float * x2h_bias,
                         float * h2h_bias,
                         float * out_state);

NN_OP_STATUS_t nn_op_norm(float * im_in,
                          float * norm_w,
                          float * norm_bias,
                          int len,
                          int seq_len,
                          float * out);

NN_OP_STATUS_t nn_op_fc(float * im_in,
                        float * w,
                        float * b,
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
