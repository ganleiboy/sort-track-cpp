#ifndef  UTILS_H
#define  UTILS_H
#include <iostream>
#include <fstream>  // 使用fstream读写文件
#include <opencv2/opencv.hpp>
#include "track.h"

using namespace std;

// 获取目录中所有文件的文件路径，可以指定文件类型
void getFilePaths(string &folder, vector<cv::String> &filepaths, string postfix, bool sort_=true);
// 从行人真值TXT文件中读取检测结果，返回以frame_id为key的字典
void getDetectResults(string &detfile, std::map<int, vector<Bbox>> &det_results);
// 字符串分割，返回vector，sep:分割符号。可应对如下情况："3,4,5" | "3,4,5," | ",3,4,5,"
void splitString(string &str, vector<int> &out, char sep);
#endif