#ifndef __TRACK_HPP__
#define __TRACK_HPP__

#include <set>
#include "Hungarian.h"
#include "KalmanTracker.h"

#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

struct Bbox
{
    float score;
    int bbox_id;  // bbox id in current frame
    Rect_<float> rect;
};

typedef struct TrackingBox
{
    int frame_id;
    int track_id;
    Rect_<float> box;
} TrackingBox;

struct TRACKER
{
// global variables for counting
#define CNUM 100 // max num. of people per frame
    // int total_frames = 0;  // 记录总帧数
    double total_time = 0.0;  // 记录总耗时

    Scalar_<int> randColor[CNUM];

    int frame_count = 0; // 记录处理了多少帧数据。由于刚调用update函数就会加一，所以是从1开始计数
    int max_age = 3;     // 连续预测的最大次数，即目标未被检测到的帧数，超过之后会被删
    int min_hits = 3;    // 目标命中的最小次数，小于该次数时update函数不返回该目标的KalmanTracker卡尔曼滤波对象
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
        for (int i = 0; i < CNUM; i++)
            rng.fill(randColor[i], RNG::UNIFORM, 0, 256);
    }

    // Computes IOU between two bounding boxes
    double GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt)
    {
        float in = (bb_test & bb_gt).area();
        float un = bb_test.area() + bb_gt.area() - in;

        if (un < DBL_EPSILON)
            return 0;

        return (double)(in / un);
    }

    vector<TrackingBox> update(const vector<TrackingBox> &detFrameData)
    {
        // total_frames++;
        frame_count++;

        // count running time using clock()
        start_time = getTickCount();

        // 初始化，the first frame met
        if (trackers.size() == 0) 
        {
            // initialize kalman trackers using first detections.
            for (unsigned int i = 0; i < detFrameData.size(); i++)
            {
                KalmanTracker trk = KalmanTracker(detFrameData[i].box);
                trackers.push_back(trk);
            }
            return vector<TrackingBox>();
        }

        ///////////////////////////////////////
        // 3.1. get predicted locations from existing trackers.
        predictedBoxes.clear();

        for (auto it = trackers.begin(); it != trackers.end();)
        {
            Rect_<float> pBox = (*it).predict();
            if (pBox.x >= 0 && pBox.y >= 0)
            {
                predictedBoxes.push_back(pBox);
                it++;
            }
            else
            {
                it = trackers.erase(it);
                //cerr << "Box invalid at frame: " << frame_count << endl;
            }
        }

        ///////////////////////////////////////
        // 3.2. associate detections to tracked object (both represented as bounding boxes)
        // dets : detFrameData[fi]
        trkNum = predictedBoxes.size();
        detNum = detFrameData.size();

        iouMatrix.clear();
        iouMatrix.resize(trkNum, vector<double>(detNum, 0));
        // compute iou matrix as a distance matrix
        for (unsigned int i = 0; i < trkNum; i++) 
        {
            for (unsigned int j = 0; j < detNum; j++)
            {
                // use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
                iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], detFrameData[j].box);
            }
        }

        // solve the assignment problem using hungarian algorithm.
        // the resulting assignment is [track(prediction) : detection], with len=preNum
        HungarianAlgorithm HungAlgo;
        assignment.clear();
        double cost_ = HungAlgo.Solve(iouMatrix, assignment);
        if (cost_ == -1.0) {
            cout << "hungarian assignment error !" << endl; // 如果是因为异常值退出，则打印
        }

        // find matches, unmatched_detections and unmatched_predictions
        unmatchedTrajectories.clear();
        unmatchedDetections.clear();
        allItems.clear();
        matchedItems.clear();

        if (detNum > trkNum) //	there are unmatched detections
        {
            for (unsigned int n = 0; n < detNum; n++)
                allItems.insert(n);

            for (unsigned int i = 0; i < trkNum; ++i)
                matchedItems.insert(assignment[i]);

            set_difference(allItems.begin(), allItems.end(),
                           matchedItems.begin(), matchedItems.end(),
                           insert_iterator<set<int>>(unmatchedDetections, unmatchedDetections.begin()));
        }
        else if (detNum < trkNum) // there are unmatched trajectory/predictions
        {
            for (unsigned int i = 0; i < trkNum; ++i)
                if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
                    unmatchedTrajectories.insert(i);
        }

        // filter out matched with low IOU
        matchedPairs.clear();
        for (unsigned int i = 0; i < trkNum; ++i)
        {
            if (assignment[i] == -1) // pass over invalid values
                continue;
            if (1 - iouMatrix[i][assignment[i]] < iouThreshold)
            {
                unmatchedTrajectories.insert(i);
                unmatchedDetections.insert(assignment[i]);
            }
            else
                matchedPairs.push_back(cv::Point(i, assignment[i]));
        }

        ///////////////////////////////////////
        // 3.3. updating trackers

        // update matched trackers with assigned detections.
        // each prediction is corresponding to a tracker
        int detIdx, trkIdx;
        for (unsigned int i = 0; i < matchedPairs.size(); i++)
        {
            trkIdx = matchedPairs[i].x;
            detIdx = matchedPairs[i].y;
            trackers[trkIdx].update(detFrameData[detIdx].box);
        }

        // create and initialise new trackers for unmatched detections
        for (auto umd : unmatchedDetections)
        {
            KalmanTracker tracker = KalmanTracker(detFrameData[umd].box);
            trackers.push_back(tracker);
        }

        // get trackers' output
        frameTrackingResult.clear();
        for (auto it = trackers.begin(); it != trackers.end();)
        {
            // min_hits不设置为0是因为第一次检测到的目标不用跟踪，不能设大，一般就是1，表示如果连续两帧都检测到目标
            int time_window = 1;  // 表示连续预测的次数
            if (((*it).m_time_since_update < time_window) &&
                ((*it).m_hit_streak >= min_hits || frame_count <= min_hits))
            {
                TrackingBox res;
                res.box = (*it).lastRect;
                res.track_id = (*it).m_id + 1;  // +1 as MOT benchmark requires positive
                res.frame_id = frame_count;
                frameTrackingResult.push_back(res);
                it++;
            }
            else
                it++;

            // remove dead tracklet
            if (it != trackers.end() && (*it).m_time_since_update > max_age)
                it = trackers.erase(it);
        }

        cycle_time = (double)(getTickCount() - start_time);
        total_time += cycle_time / getTickFrequency();

        return frameTrackingResult;
    }
};

#endif