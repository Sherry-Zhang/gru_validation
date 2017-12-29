 /**
  * @file      test_gru_forward.cpp
  * @author    zhangshu(shu.zhang@intel.com)
  * @date      2017-12-20 09:07:22
  * @brief
  **/

#include <cstdlib>
#include <math.h>
#include <vector>
#include <map>
#include <iostream>
#include "utils.hpp"
using namespace std;
template <typename data_t>
void grucell_fwd(const int T,
                 const int N,
                 const int I,
                 const int H,
                 bool bid,
                 const data_t* x,   //[T, N, I]
                 const data_t* hx,  //[N, H]
                 const data_t* weights,
                 data_t* y,         //[T, N, H]
                 data_t* hy) {      //[N, H]
    const data_t *wx = weights;  // 
    const data_t *wh = wx + I * H * 3;
    const data_t *bx = wh + H * H * 3;
    const data_t *bh = bx + H * 3;
    //need workspace to save [rt|zt|nt], size:[N, 3H]
    const int col = H * 3;
    data_t *buf = new data_t[N * col];
    data_t (*rt)[col] = (data_t(*)[col])buf;
    data_t (*zt)[col] = (data_t(*)[col])((data_t*)rt + H);
    data_t (*nt)[col] = (data_t(*)[col])((data_t*)zt + H);

    data_t *hbuf = new data_t[N * col];
    data_t (*rh)[col] = (data_t(*)[col])hbuf;
    data_t (*zh)[col] = (data_t(*)[col])((data_t*)rh + H);
    data_t (*nh)[col] = (data_t(*)[col])((data_t*)zh + H);

    const data_t *brx = bx;
    const data_t *bzx = brx + H;
    const data_t *bnx = bzx + H;
    const data_t *brh = bh;
    const data_t *bzh = brh + H;
    const data_t *bnh = bzh + H;
     
    const data_t *xt = x;
    const data_t *ht_pre = hx;
    //t0 --> tn
    if(!bid) {
        for (int i = 0; i < T; ++i) {
           gemm<data_t>(NOTRANS, NOTRANS,     xt, wx,  buf, N, col, I, 0);
           gemm<data_t>(NOTRANS, NOTRANS, ht_pre, wh, hbuf, N, col, H, 0); 
           for (int j = 0; j < N; ++j) {
               int row = i * N + j;
               for (int k = 0; k < H; ++k) {
                   rt[j][k] = sigmoid(rt[j][k] + brx[k] + rh[j][k] + brh[k]);
                   zt[j][k] = sigmoid(zt[j][k] + bzx[k] + zh[j][k] + bzh[k]);
                   nt[j][k] =    tanh(nt[j][k] + bnx[k] + rt[j][k] * (nh[j][k] + bnh[k]));
                   hy[j * H + k] = (1 - zt[j][k]) * nt[j][k] + zt[j][k] * ht_pre[j * H + k];
                   y[row * H + k] = hy[j * H + k];
               }
           }
           xt += N * I;
           ht_pre = hy;
        }
    }
    //tn --> t0
    else {
        xt = x + (T - 1) * N * I;
        for (int i = T - 1; i >= 0; --i) {
           gemm<data_t>(NOTRANS, NOTRANS,     xt, wx,  buf, N, col, I, 0);
           gemm<data_t>(NOTRANS, NOTRANS, ht_pre, wh, hbuf, N, col, H, 0); 
           for (int j = 0; j < N; ++j) {
               int row = i * N + j;
               for (int k = 0; k < H; ++k) {
                   rt[j][k] = sigmoid(rt[j][k] + brx[k] + rh[j][k] + brh[k]);
                   zt[j][k] = sigmoid(zt[j][k] + bzx[k] + zh[j][k] + bzh[k]);
                   nt[j][k] =    tanh(nt[j][k] + bnx[k] + rt[j][k] * (nh[j][k] + bnh[k]));
                   hy[j * H + k] = (1 - zt[j][k]) * nt[j][k] + zt[j][k] * ht_pre[j * H + k];
                   y[row * H + k] = hy[j * H + k];
               }
           }
           xt -= N * I;
           ht_pre = hy;
        }
    }
    delete []buf;
    delete []hbuf;
}

/**
 * @brief: The test function to compute gru_forward.
 */
template <typename data_t>
void compute_ref_gru_fwd(
        const int T,
        const int N,
        const int I,
        const int H,
        const int nd,
        const int nl,
        const data_t* x,
        const data_t* hx,
        const data_t* weights,
        data_t* y,
        data_t* hy) {
    const data_t *x_ptr = x; 
    const data_t *hx_ptr = hx;
    const data_t *weights_ptr = weights;
    data_t *y_ptr = y; 
    data_t *hy_ptr = hy; 

    data_t *reorder_y = y_ptr;  //if bidirect, need a tmp memory to save reorder y
    data_t *y_bid = NULL;
    if (nd == BIDIRECT) {
        reorder_y = new data_t[T * N * H * nd];
        y_bid = reorder_y + T * N * H;
    }
    for (int l = 0; l < nl; ++l) {
        int input_size = (l == 0) ? I : nd * H;
        grucell_fwd(T, N, input_size, H, false, x_ptr, hx_ptr, weights_ptr, reorder_y, hy_ptr);
        hx_ptr += N * H; 
        weights_ptr +=  (input_size + H + 2) * H * 3;
        hy_ptr += N * H;
        if (nd == BIDIRECT) {
            grucell_fwd(T, N, input_size, H, true, x_ptr, hx_ptr, weights_ptr, y_bid, hy_ptr);
            hx_ptr += N * H; 
            weights_ptr +=  (input_size + H + 2) * H * 3;
            hy_ptr += N * H;
            //y_ptr:[T, N, H * nd], reorder_y:[nd * T, N, H]
            for (int i = 0; i < T; ++i) {
                for (int j = 0; j < N; ++j) {
                    int row = i * N + j;
                    for (int k = 0; k < H; ++k) {
                        y_ptr[row * H * nd + k] = reorder_y[row * H + k];
                    }
                    for (int k = 0; k < H; ++k) {
                        y_ptr[row * H * nd + H + k] = reorder_y[(row + T * N) * H + k]; 
                    }
                }
            }
        }
        x_ptr = y_ptr;
    }
    if (nd == BIDIRECT) {
        delete []reorder_y;
    }
}

int main(int argc, char* argv[])
{
    const char* file_path = "../tmp/gru.json";
    std::map<std::string, std::string> data_map;
    load_json(data_map, file_path);
    const int T = atoi(data_map["seq_length"].c_str()); 
    const int N = atoi(data_map["batch_size"].c_str());
    const int I = atoi(data_map["input_size"].c_str());
    const int H = atoi(data_map["hidden_size"].c_str());
    const int nd = atoi(data_map["num_direction"].c_str());
    const int nl = atoi(data_map["num_layer"].c_str());
    //allocate memory
    float* x = new float[T * N * I];
    float* hx = new float[nl * nd * N * H];
    const int size1 = (I + H + 2) * H * 3 * nd;    //first layer
    const int size2 = (nd*H + H + 2) * H * 3 * nd;    //other layers
    float* weights = new float[size1 + (nl - 1) * size2];
    float* hy = new float[nl * nd * N * H];
    float* y = new float[T * N * H * nd];
    float* ref_y = new float[T * N * H * nd];
    float* ref_hy = new float[nl * nd * N * H];
    //fill_data
    std::vector<std::string> s;
    split(s, data_map["weights"], " ");
    my_fill_data(s, weights);
    split(s, data_map["x"], " ");
    my_fill_data(s, x);
    split(s, data_map["hx"], " ");
    my_fill_data(s, hx);
    split(s, data_map["y"], " ");
    my_fill_data(s, y);
    split(s, data_map["hy"], " ");
    my_fill_data(s, hy);
    compute_ref_gru_fwd<float>(T, N, I, H, nd, nl, x, hx, weights, ref_y, ref_hy);
    if (compare_mem(ref_y, y, T * N * H * nd, 0.0001f)) {
        cout << "check y:   Pass" << endl;
    }
    else {
        cout << "check y:   Fail" << endl;
    }
    if (compare_mem(ref_hy, hy, nl * nd * N * H, 0.0001f)) {
        cout << "check hy:  Pass" << endl;
    }
    else {
        cout << "check hy:  Fail" << endl;
    }
    delete []x;
    delete []hx;
    delete []weights;
    delete []hy;
    delete []y;
    delete []ref_y;
    delete []ref_hy;
    return 0;
}
