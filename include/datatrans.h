#ifndef __DATATRANS_H
#define __DATATRANS_H

#include <opencv2/opencv.hpp>

using namespace std;

struct Bbox
{
    float score;
    int class_id;  // 类别ID
    int bbox_id;  // bbox id in current frame
    cv::Rect_<float> rect;
};


struct TrackingBox
{
    int frame_id;
    int track_id;
    int class_id;
    float obj_conf;  // 是否为前景的置信度
    cv::Rect_<float> box;

    // 构造函数
    TrackingBox(){}  
    // 重载构造函数
    TrackingBox(Bbox obj){
        box = obj.rect;
        obj_conf = obj.score;
        class_id = obj.class_id;
        track_id = -1;  // 初始化为-1
    }
};

#endif // DATATRANS_H