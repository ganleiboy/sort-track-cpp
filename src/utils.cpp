#include "utils.h"

void getFilePaths(string &folder, vector<cv::String> &filepaths, string postfix, bool sort_){
    // 获取目录中某种格式的文件，返回其路径
    
    // 1，根据后缀进行过滤
    vector<cv::String> res;
    cv::glob(folder, res);  // 返回值是路径
    for (auto filepath:res){
        // 获取字符串文件名的后缀
        if (filepath.substr(filepath.find_last_of('.') + 1) == postfix){
            filepaths.push_back(filepath);
        }
    }
    // 2，排序
    if (sort_){
        sort(filepaths.begin(), filepaths.end());
    }
}

void getDetectResults(string &detfile, std::map<int, vector<Bbox>> &det_results){
    ifstream fin;
    fin.open(detfile, std::ios::in);
    string str_line;
    // 逐行读取
    while (getline(fin, str_line))
    {
        vector<int> res;  // 当前行按逗号分割后的结果
        splitString(str_line, res, ',');
        // 解析bbox
        int frame_id = res[0];
        Bbox bbox;
        bbox.bbox_id = res[1];
        bbox.rect = cv::Rect_<int>(res[2], res[3], res[4], res[5]);
        // 判断map中某个key是否存在，如果不存在需要先初始化
        if (det_results.find(frame_id) == det_results.end()){
            det_results[frame_id] = vector<Bbox>();  // 初始化为空值
        }
        det_results[frame_id].push_back(bbox);
    }
}

void splitString(string &str, std::vector<int> &out, char sep)
{
    string::size_type start = str.find_first_not_of(sep, 0); // 找到第一个不为逗号的下标
    string::size_type pose = str.find_first_of(sep, start);  // 找到第一个逗号的下标
    while (string::npos != start || string::npos != pose)
    { // 当即没有逗号也没有字符的时候结束
        out.push_back(atoi(str.substr(start, pose - start).c_str()));  // 字符串转int
        start = str.find_first_not_of(sep, pose); // 更新start 从pose开始
        pose = str.find_first_of(sep, start);     // 更新pos,从start开始
    }
}
