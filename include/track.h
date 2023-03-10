#ifndef __TRACK_H__
#define __TRACK_H__

#include <set>
#include "Hungarian.h"
#include "KalmanTracker.h"
#include "datatrans.h"

#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;


class TRACKER
{
public:
    // int total_frames = 0;  // 记录总帧数
    double total_time = 0.0;  // 记录总耗时
    static const int max_num = 100;  // max num of people per frame
    Scalar_<int> randColor[max_num];  // 颜色数组

    int frame_count = 0;         // 记录处理了多少帧数据。由于刚调用update函数就会加一，所以实际是从1开始计数
    int max_lost_time = 3;       // 连续预测的最大次数，即目标未被检测到的帧数，超过之后会被删
    int lower_max_lost_time = 2; // 对新生tracker和稳定tracker的容忍度不同，如果稳定tracker连续丢失3次就被移除，那新生tracker则是2次
    int min_hits = 3;            // 目标命中的最小次数，小于该次数时getReport函数不返回该目标的KalmanTracker卡尔曼滤波对象
    double iouThreshold = 0.3;
    vector<KalmanTracker> trackers; // 维护所有的跟踪序列，列表元素是KalmanTracker的对象

    // variables used in the for-loop
    vector<Rect_<float>> predictedBoxes;
    vector<vector<double>> iouMatrix;
    vector<int> assignment;
    set<int> unmatchedDetections;
    set<int> unmatchedTrajectories;
    set<int> allItems;
    set<int> matchedItems;
    vector<cv::Point> matchedPairs;
    vector<TrackingBox> frameTrackingResult;  // 用于保存最新的对外输出结果
    unsigned int trkNum = 0;
    unsigned int detNum = 0;

    double cycle_time = 0.0;
    int64 start_time = 0;

    TRACKER()
    {
        KalmanTracker::kf_count = 0; // tracking id relies on this, so we have to reset it in each seq.
        RNG rng(0xFFFFFFFF);
        for (int i = 0; i < max_num; i++)
            rng.fill(randColor[i], RNG::UNIFORM, 0, 256);
    }

    // Computes IOU between two bounding boxes
    double getIOU(Rect_<float> bb_test, Rect_<float> bb_gt);
    // 更新全局跟踪器
    void update(const vector<TrackingBox> &detFrameData);
    // 对外输出跟踪结果
    vector<TrackingBox> getReport();
};

#endif