 /**
  * @file      utils.hpp
  * @author    zhangshu(shu.zhang@intel.com)
  * @date      2017-12-29 14:16:33
  * @brief
  **/
#ifndef UTILS_HPP
#define UTILS_HPP

enum { UNIDIRECT = 1, BIDIRECT = 2 };
enum { NOTRANS = 1, TRANS = 2 };

int load_json(std::map<std::string, std::string> &data_map, const char* file_path);
void split(std::vector<std::string> &out, std::string &str, std::string delim);
void print(const float *array, int time_step, int row, int col);
void my_fill_data(std::vector<std::string> &s, float* ptr);
bool compare_mem(float* a, float* b, const int len, float diff);

template <typename data_t>
void gemm(const int transA, const int transB, const data_t *A,
        const data_t *B, data_t *C, const int M, const int N, const int K,
        const data_t beta)
{
    int m, n, k;
    if (beta == 0) {
        for (m = 0; m < M * N; m++) {
            C[m] = static_cast<data_t>(0.);
        }
    }
    for (k = 0; k < K; k++) {
        for (m = 0; m < M; m++) {
            for (n = 0; n < N; n++) {
                if (transA == NOTRANS && transB == NOTRANS)
                    C[m * N + n] += A[m * K + k] * B[k * N + n];
                if (transA == TRANS && transB == NOTRANS)
                    C[m * N + n] += A[k * M + m] * B[k * N + n];
                if (transA == NOTRANS && transB == TRANS)
                    C[m * N + n] += A[m * K + k] * B[n * K + k];
                if (transA == TRANS && transB == TRANS)
                    C[m * N + n] += A[k * M + m] * B[n * K + k];
            }
        }
    }
}
#endif // UTILS_HPP
