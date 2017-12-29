 /**
  * @file      test_gru_backward.cpp
  * @author    zhangshu(shu.zhang@intel.com)
  * @date      2017-12-25 13:27:29
  * @brief
  **/
#include <cstdlib>
#include <cstring>
#include <cmath>
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
                 data_t* reserve_buf, //[T, N,3 * H] + [T, N, H] + [T, N, H]
                 const data_t* x,   //[T, N, I]
                 const data_t* hx,  //[N, H]
                 const data_t* weights,
                 data_t* y,         //[T, N, H]
                 data_t* hy) {      //[N, H]
    const data_t *wx = weights;  // 
    const data_t *wh = wx + I * H * 3;
    const data_t *bx = wh + H * H * 3;
    const data_t *bh = bx + H * 3;

    const data_t *brx = bx;
    const data_t *bzx = brx + H;
    const data_t *bnx = bzx + H;
    const data_t *brh = bh;
    const data_t *bzh = brh + H;
    const data_t *bnh = bzh + H;

    const int col = H * 3;
    data_t *rt = reserve_buf;
    data_t *zt = rt + H;
    data_t *nt = zt + H;
    data_t *n_gemm = reserve_buf + T * N * H * 3;
    data_t *ht = n_gemm + T * N * H;

    data_t *hbuf = new data_t[N * col];
     
    const data_t *xt = x;
    const data_t *ht_pre = hx;
    //t0 --> tn
    if (!bid) {
        for (int i = 0; i < T; ++i) {
            gemm<data_t>(NOTRANS, NOTRANS,     xt, wx,   rt, N, col, I, 0);
            gemm<data_t>(NOTRANS, NOTRANS, ht_pre, wh, hbuf, N, col, H, 0); 
            data_t *rh = hbuf;
            data_t *zh = rh + H;
            data_t *nh = zh + H;
            for (int j = 0; j < N; ++j) {
                int row = i * N + j;
                for (int k = 0; k < H; ++k) {
                    rt[k] = sigmoid(rt[k] + brx[k] + rh[k] + brh[k]);
                    zt[k] = sigmoid(zt[k] + bzx[k] + zh[k] + bzh[k]);
                    nt[k] =    tanh(nt[k] + bnx[k] + rt[k] * (nh[k] + bnh[k]));
                    n_gemm[k] = nh[k] + bnh[k];
                    ht[k] = (1 - zt[k]) * nt[k] + zt[k] * ht_pre[j * H + k];
                    y[row * H + k] = hy[j * H + k] = ht[k];
                }
                rt += col;
                zt += col;
                nt += col;
                n_gemm += H;
                ht += H;
                rh += col;
                zh += col;
                nh += col;
            }
            xt += N * I;
            ht_pre = hy;
        }
    }
    //tn --> t0
    else {
        for (int i = T - 1; i >= 0; --i) {
            xt = x + i * N * I;
            rt = reserve_buf + i * N * H * 3; 
            zt = rt + H;
            nt = zt + H;
            n_gemm = reserve_buf + T * N * H * 3 + i * N * H;
            ht = reserve_buf + T * N * H * 4 + i * N * H;
            gemm<data_t>(NOTRANS, NOTRANS,     xt, wx,   rt, N, col, I, 0);
            gemm<data_t>(NOTRANS, NOTRANS, ht_pre, wh, hbuf, N, col, H, 0); 
            data_t *rh = hbuf;
            data_t *zh = rh + H;
            data_t *nh = zh + H;
            for (int j = 0; j < N; ++j) {
                int row = i * N + j;
                for (int k = 0; k < H; ++k) {
                    rt[k] = sigmoid(rt[k] + brx[k] + rh[k] + brh[k]);
                    zt[k] = sigmoid(zt[k] + bzx[k] + zh[k] + bzh[k]);
                    nt[k] =    tanh(nt[k] + bnx[k] + rt[k] * (nh[k] + bnh[k]));
                    n_gemm[k] = nh[k] + bnh[k];
                    ht[k] = (1 - zt[k]) * nt[k] + zt[k] * ht_pre[j * H + k];
                    y[row * H + k] = hy[j * H + k] = ht[k];
                }
                rt += col;
                zt += col;
                nt += col;
                n_gemm += H;
                ht += H;
                rh += col;
                zh += col;
                nh += col;
            }
            xt -= N * I;
            ht_pre = hy;
        }
    }
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
        data_t* reserve_buf,  //nl*nd*5*T*N*H
        const data_t* x,
        const data_t* hx,       
        const data_t* weights,
        data_t* y,
        data_t* hy) {
    const data_t *x_ptr = x; 
    const data_t *hx_ptr = hx;
    const data_t *weights_ptr = weights;
    data_t *reserve_buf_ptr = reserve_buf;
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
        grucell_fwd(T, N, input_size, H, false, reserve_buf_ptr, x_ptr, hx_ptr, weights_ptr, reorder_y, hy_ptr);
        hx_ptr += N * H; 
        weights_ptr +=  (input_size + H + 2) * H * 3;
        hy_ptr += N * H;
        reserve_buf_ptr += T * N * H * 5;
        if (nd == BIDIRECT) {
            grucell_fwd(T, N, input_size, H, true, reserve_buf_ptr, x_ptr, hx_ptr, weights_ptr, y_bid, hy_ptr);
            hx_ptr += N * H; 
            weights_ptr +=  (input_size + H + 2) * H * 3;
            hy_ptr += N * H;
            reserve_buf_ptr += T * N * H * 5;
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

template <typename data_t>
void grucell_bwd(const int T,
                 const int N,
                 const int I,
                 const int H,
                 bool bid,
                 const data_t* reserve_buf, //[T * N * H * 4]
                 const data_t* x,   //[T, N, I]
                 const data_t* hx,  //[N, H]
                 const data_t* weights,
                 const data_t* hy, 
                 const data_t* grad_y,
                 const data_t* grad_hy,
                 data_t* grad_x,
                 data_t* grad_hx,
                 data_t* grad_weights) {      //[N, H]
    const data_t *wx = weights;  // 
    const data_t *wh = wx + I * H * 3;

    const int col = H * 3;

    data_t *deta_wx = grad_weights;  // 
    data_t *deta_wh = deta_wx + I * H * 3;
    data_t *deta_bx = deta_wh + H * H * 3;
    data_t *deta_bh = deta_bx + H * 3;

    data_t *deta_brx = deta_bx;
    data_t *deta_bzx = deta_brx + H;
    data_t *deta_bnx = deta_bzx + H;
    data_t *deta_brh = deta_bh;
    data_t *deta_bzh = deta_brh + H;
    data_t *deta_bnh = deta_bzh + H;

    data_t *buf = new data_t[N * col];      //grad_buf [dr|dz|dn]
    
    const data_t *xt = x + (T - 1) * N * I;
    const data_t *y = reserve_buf + T * N * H * 4;
    data_t *deta_h = new data_t[T * N * H];
    memcpy(deta_h, grad_y, T * N * H * sizeof(data_t));

    if (!bid) {
        for (int i = T - 1; i >= 0; --i) {
            const data_t *hnext = i ? y + (i - 1) * N * H: hx;
            data_t *deta_hnext = i ? deta_h + (i - 1) * N * H: grad_hx;
            const data_t *rt = reserve_buf + i * N * H * 3;
            const data_t *zt = rt + H;
            const data_t *nt = zt + H;
            const data_t *nh = reserve_buf + T * N * col + i * N * H;

            data_t *deta_r = buf;
            data_t *deta_z = deta_r + H;
            data_t *deta_n = deta_z + H;
            for (int j = 0; j < N; ++j) {
                int row = i * N + j;
                for (int k = 0; k < H; ++k) {
                    if (i == T - 1) {
                        deta_h[row * H + k] += grad_hy[j * H + k];
                    }
                    deta_n[k] = deta_h[row * H + k] * (1 - zt[k]) * (1 - nt[k] * nt[k]);
                    deta_z[k] = deta_h[row * H + k] * (hnext[j * H + k] - nt[k]) * zt[k] * (1 - zt[k]);
                    deta_r[k] = deta_n[k] * nh[k] * rt[k] * (1 - rt[k]);
                    deta_hnext[j * H + k] += deta_h[row * H + k] * zt[k];
                    deta_brx[k] += deta_r[k];
                    deta_bzx[k] += deta_z[k];
                    deta_bnx[k] += deta_n[k];
                    deta_brh[k] += deta_r[k];
                    deta_bzh[k] += deta_z[k];
                    deta_bnh[k] += deta_n[k] * rt[k];
                }
                rt += col;
                zt += col;
                nt += col;
                nh += H;
                deta_r += col;
                deta_z += col;
                deta_n += col;
            }
            //dx, dwx
            gemm<data_t>(TRANS, NOTRANS, xt, buf, deta_wx, I, col, N, 1);
            gemm<data_t>(NOTRANS, TRANS, buf, wx, grad_x + i * N * I, N, I, col, 0);

            //update deta_n and compute dwh, dhnext
            rt = reserve_buf + i * N * H * 3;
            deta_n = buf + H + H;
            for (int j = 0; j < N; ++j) {
                for (int k = 0; k < H; ++k) {
                    deta_n[k] *= rt[k];
                }
                rt += col;
                deta_n += col;
            }
            gemm<data_t>(TRANS, NOTRANS, hnext, buf, deta_wh, H, col, N, 1);
            gemm<data_t>(NOTRANS, TRANS, buf, wh, deta_hnext, N, H, col, 1);
            xt -= N * I;
        }
    }
    // tn --> t0
    else {
        const data_t *rt = reserve_buf;
        const data_t *zt = rt + H;
        const data_t *nt = zt + H;
        const data_t *nh = reserve_buf + T * N * col;
        xt = x;
        for (int i = 0; i < T; ++i) {
            const data_t *hnext = (i != T - 1) ? y + (i + 1) * N * H: hx;
            data_t *deta_hnext = (i != T - 1) ? deta_h + (i + 1) * N * H: grad_hx;
            data_t *deta_r = buf;
            data_t *deta_z = deta_r + H;
            data_t *deta_n = deta_z + H;
            for (int j = 0; j < N; ++j) {
                int row = i * N + j;
                for (int k = 0; k < H; ++k) {
                    if (i == 0) {
                        deta_h[row * H + k] += grad_hy[j * H + k];
                    }
                    deta_n[k] = deta_h[row * H + k] * (1 - zt[k]) * (1 - nt[k] * nt[k]);
                    deta_z[k] = deta_h[row * H + k] * (hnext[j * H + k] - nt[k]) * zt[k] * (1 - zt[k]);
                    deta_r[k] = deta_n[k] * nh[k] * rt[k] * (1 - rt[k]);
                    deta_hnext[j * H + k] += deta_h[row * H + k] * zt[k];
                    deta_brx[k] += deta_r[k];
                    deta_bzx[k] += deta_z[k];
                    deta_bnx[k] += deta_n[k];
                    deta_brh[k] += deta_r[k];
                    deta_bzh[k] += deta_z[k];
                    deta_bnh[k] += deta_n[k] * rt[k];
                }
                rt += col;
                zt += col;
                nt += col;
                nh += H;
                deta_r += col;
                deta_z += col;
                deta_n += col;
            }
            //dx, dwx
            gemm<data_t>(TRANS, NOTRANS, xt, buf, deta_wx, I, col, N, 1);
            gemm<data_t>(NOTRANS, TRANS, buf, wx, grad_x + i * N * I, N, I, col, 1);

            //update deta_n and compute dwh, dhnext
            rt = reserve_buf + i * N * H * 3;
            deta_n = buf + H + H;
            for (int j = 0; j < N; ++j) {
                for (int k = 0; k < H; ++k) {
                    deta_n[k] *= rt[k];
                }
                rt += col;
                deta_n += col;
            }
            gemm<data_t>(TRANS, NOTRANS, hnext, buf, deta_wh, H, col, N, 1);
            gemm<data_t>(NOTRANS, TRANS, buf, wh, deta_hnext, N, H, col, 1);
            xt += N * I;
        }
    }
    delete []buf;
}
/**
 * @brief       The test function of gru backward computation
 * @params      T: seq_length
 *              N: batch_size
 *              I: input_size
 *              H: hidden_size
 *              nd: num_direction
 *              nl: num_layer
 *              reserve_buf: the buffer is used for computing 
 *                           intermediate values during forward pass, 
 *                           including rt, zt, nt, nh and ht
 *              x:  [T, N, I] input features             
 *              hx: [nl * nd, N, H] initial hidden state
 *              weights: nl * nd * single_w_size
 *              y: 
 *
 */
template <typename data_t>
void compute_ref_gru_bwd(
        const int T,
        const int N,
        const int I,
        const int H,
        const int nd,
        const int nl,
        const data_t* reserve_buf, //nl*nd*5*T*N*H
        const data_t* x,
        const data_t* hx,       
        const data_t* weights,
        const data_t* hy,
        data_t* grad_y,
        const data_t* grad_hy,
        data_t* grad_x,
        data_t* grad_hx,
        data_t* grad_weights) {
    const int size1 = (I + H + 2) * H * 3 * nd;    //first layer
    const int size2 = (nd * H + H + 2) * H * 3 * nd;    //other layers
    data_t* tmp = new float[T * N * H * nd];
    data_t* xnext = NULL;
    if (nd == BIDIRECT) {
        xnext = new float[T * N * H * nd];
    }
    for (int l = nl - 1; l >= 0; --l) {
        int input_size = l ? nd * H : I;
        const data_t *reserve_buf_ptr = reserve_buf +
                                        l * nd * T * N * H * 5;
        const data_t *hy_ptr = hy + l * nd * N * H; 
        const data_t *hx_ptr = hx + l * nd * N * H;
        const data_t *weights_ptr = 
                    l ? weights + size1 + (l - 1) * size2 : weights;
        const data_t *grad_hy_ptr = grad_hy + l * nd * N * H;
        data_t *grad_x_ptr = l ? tmp : grad_x;
        data_t *grad_hx_ptr = grad_hx + l * nd * N * H;
        data_t *grad_weights_ptr = 
                    l ? grad_weights + size1 + (l - 1) * size2 : grad_weights;
        if (l != nl - 1) {
            memcpy(grad_y, tmp, T * N * H * nd * sizeof(data_t));
        }
        if (nd == BIDIRECT) {
            //1.reorder grad_y from [T, N, H * nd] to [nd * T, N, H]
            for (int i = 0; i < T; ++i) {
                for (int j = 0; j < N; ++j) {
                    int row = i * N + j;
                    for (int k = 0; k < H; ++k) {
                        tmp[row * H + k] = grad_y[row * H * nd + k];
                    }
                    for (int k = 0; k < H; ++k) {
                        tmp[(row + T * N) * H + k] = grad_y[row * H * nd + H + k];
                    }
                }
            }
            memcpy(grad_y, tmp, T * N * H * nd * sizeof(data_t));
            const data_t *x_ptr = l ? xnext : x;
            //2.fill xnext
            if (l) {
                const data_t *ynext = reserve_buf_ptr - T * N * H * 6;
                const data_t *ynext_bid = reserve_buf_ptr - T * N * H; 
                for (int i = 0; i < T; ++i) {
                    for (int j = 0; j < N; ++j) {
                        int row = i * N + j;
                        for (int k = 0; k < H; ++k) {
                            xnext[row * H * nd + k] = ynext[row * H + k];
                        }
                        for (int k = 0; k < H; ++k) {
                            xnext[row * H * nd + H + k] = ynext_bid[row * H + k];
                        }
                    }
                }
            }
            grucell_bwd(T, N, input_size, H, false, reserve_buf_ptr, x_ptr, hx_ptr, 
                        weights_ptr, hy_ptr, grad_y, grad_hy_ptr, 
                        grad_x_ptr, grad_hx_ptr, grad_weights_ptr);

            reserve_buf_ptr += T * N * H * 5;
            hy_ptr += N * H;
            hx_ptr += N * H;
            weights_ptr += (input_size + H + 2) * H * 3;
            grad_hy_ptr += N * H;
            grad_hx_ptr += N * H;
            grad_weights_ptr += (input_size + H + 2) * H * 3;
            grucell_bwd(T, N, input_size, H, true, reserve_buf_ptr, x_ptr, 
                        hx_ptr, weights_ptr, hy_ptr, grad_y + T * N * H, 
                        grad_hy_ptr, grad_x_ptr, grad_hx_ptr, grad_weights_ptr);
        }
        else {
            const data_t *x_ptr = l ? reserve_buf_ptr - T * N * H : x;
            grucell_bwd(T, N, input_size, H, false, reserve_buf_ptr, x_ptr, hx_ptr, 
                        weights_ptr, hy_ptr, grad_y, grad_hy_ptr, 
                        grad_x_ptr, grad_hx_ptr, grad_weights_ptr);
        }
    }
    delete []tmp;
    if (nd == BIDIRECT) {
        delete []xnext;
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
// allocate memory
    float* x = new float[T * N * I];
    float* hx = new float[nl * nd * N * H];
    const int size1 = (I + H + 2) * H * 3 * nd;    //first layer
    const int size2 = (nd*H + H + 2) * H * 3 * nd;    //other layers
    float* weights = new float[size1 + (nl - 1) * size2];
    // forward
    float* hy = new float[nl * nd * N * H];
    float* y = new float[T * N * H * nd];
    float* ref_y = new float[T * N * H * nd];
    float* ref_hy = new float[nl * nd * N * H];
    // backward
    float* reserve_buf = new float[nl * nd * T * N * H * 5];
    float* grad_hy = new float[nl * nd * N * H];
    float* grad_y = new float[T * N * H * nd];
    float* grad_x = new float[T * N * I];
    float* grad_hx = new float[nl * nd * N * H];
    float* grad_weights = new float[size1 + (nl - 1) * size2];
    float* ref_grad_x = new float[T * N * I]();
    float* ref_grad_hx = new float[nl * nd * N * H]();
    float* ref_grad_weights = new float[size1 + (nl - 1) * size2]();
    // fill_data
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
    split(s, data_map["grad_y"], " ");
    my_fill_data(s, grad_y);
    split(s, data_map["grad_hy"], " ");
    my_fill_data(s, grad_hy);
    split(s, data_map["grad_x"], " ");
    my_fill_data(s, grad_x);
    split(s, data_map["grad_hx"], " ");
    my_fill_data(s, grad_hx);
    split(s, data_map["grad_weights"], " ");
    my_fill_data(s, grad_weights);

    compute_ref_gru_fwd<float>(T, N, I, H, nd, nl, reserve_buf, 
                               x, hx, weights, ref_y, ref_hy);
    cout <<     "---------test forward--------" << endl;
    if (compare_mem(ref_y, y, T * N * H * nd, 0.0001f)) {
        cout << "|check y:               Pass|" << endl;
    } else {
        cout << "|check y:               Fail|" << endl;
    }
    if (compare_mem(ref_hy, hy, nl * nd * N * H, 0.0001f)) {
        cout << "|check hy:              Pass|" << endl;
    } else {
        cout << "|check hy:              Fail|" << endl;
    }
    compute_ref_gru_bwd<float>(T, N, I, H, nd, nl, reserve_buf, x, hx, weights,
                               ref_hy, grad_y, grad_hy, ref_grad_x,
                               ref_grad_hx, ref_grad_weights);
    cout <<     "|-------test backward-------|" << endl;
    if (compare_mem(ref_grad_x, grad_x, T * N * I, 0.0001f)) {
        cout << "|check grad_x:          Pass|" << endl;
    } else {
        cout << "|check grad_x:          Fail|" << endl;
    }
    if (compare_mem(ref_grad_hx, grad_hx, nl * nd * N * H, 0.0001f)) { 
        cout << "|check grad_hx:         Pass|" << endl;
    } else {
        cout << "|check grad_hx:         Fail|" << endl;
    }
    if (compare_mem(grad_weights, ref_grad_weights, size1 + (nl - 1) * size2, 0.0001f)) { 
        cout << "|check grad_weights:    Pass|" << endl;
    } else {
        cout << "|check grad_weights:    Fail|" << endl;
    }
        cout << "-----------------------------" << endl;
    delete []x;
    delete []hx;
    delete []weights;
    delete []hy;
    delete []y;
    delete []ref_y;
    delete []ref_hy;
    delete []grad_y;
    delete []grad_hy;
    delete []grad_x;
    delete []grad_hx;
    delete []grad_weights;
    delete []ref_grad_x;
    delete []ref_grad_hx;
    delete []ref_grad_weights;
    return 0;
}
