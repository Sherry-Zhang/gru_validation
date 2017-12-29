 /**
  * @file      utils.cpp
  * @author    zhangshu(shu.zhang@intel.com)
  * @date      2017-12-11 17:08:28
  * @brief
  **/
#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/filewritestream.h"
#include <map>
#include <vector>
#include <cstdio>
#include <cmath>
using namespace rapidjson;

const int BUF_SIZE=6553600;

//load json file and transfer into map
int load_json(std::map<std::string, std::string> &data_map, const char* file_path)
{
    FILE *fp = fopen(file_path, "r");
    if(fp == NULL) {
        printf("Can't find %s\n", file_path);
        return -1;
    }
    //read file
    char buf[BUF_SIZE];
    FileReadStream is(fp, buf, sizeof(buf));
    Document d;
    d.ParseStream(is); 
    assert(d.IsObject());

    data_map.clear();
    for (Value::ConstMemberIterator itr = d.MemberBegin(); itr != d.MemberEnd(); ++itr) {
        std::string key = itr->name.GetString();
        const Value& val = itr->value;
        data_map.insert(make_pair(key, val.GetString()));
    }
    fclose(fp); 
    return 0;
}

void split(std::vector<std::string> &out, std::string &str, std::string delim)
{
    out.clear();
    std::string s(str);
    size_t pos;
    while((pos = s.find(delim)) != std::string::npos) {
        out.push_back(s.substr(0, pos));
        s = s.substr(pos+delim.size(), s.size());
    }
    out.push_back(s);
}

void print(const float *array, int time_step, int row, int col)
{
    int i, j, k;
    printf("%dx%dx%d\n", time_step, row, col);
    for(i = 0; i < time_step; ++i)
    {
        printf("---------\n");
        for(j = 0; j < row; ++j)
        {
            for(k = 0; k < col; ++k)
            {
                printf("%.4f ", array[i * row * col + j * col + k]);
            }
            printf("\n");
        }
        printf("\n");
    }

}
void my_fill_data(std::vector<std::string> &s, float* ptr) {
    size_t size = s.size();
    for(size_t i = 0; i < size; ++i) {
        ptr[i] = atof(s[i].c_str());
    }
}
bool compare_mem(float* a, float* b, const int len, float diff) {
    for(int i = 0; i < len; ++i) {
        if(fabs(a[i]-b[i]) > diff) {
            return false;
        }
    }
    return true;
}

