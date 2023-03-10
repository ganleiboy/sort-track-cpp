#include "track.h"

double TRACKER::getIOU(Rect_<float> bb_test, Rect_<float> bb_gt)
{
    float intersection = (bb_test & bb_gt).area();
    float unionArea = bb_test.area() + bb_gt.area() - intersection;
    if (unionArea < DBL_EPSILON)
        return 0;
    return (double)(intersection / unionArea);
}


void TRACKER::update(const vector<TrackingBox> &detFrameData)
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
            KalmanTracker trk = KalmanTracker(detFrameData[i]);
            trackers.push_back(trk);
        }
        return;
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
            iouMatrix[i][j] = 1 - getIOU(predictedBoxes[i], detFrameData[j].box);
        }
    }

    // solve the assignment problem using hungarian algorithm.
    // the resulting assignment is [track(prediction) : detection], with len=preNum
    HungarianAlgorithm HungAlgo;
    assignment.clear();
    double cost_ = HungAlgo.Solve(iouMatrix, assignment);
    if (cost_ == -1.0)
    {
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
        // 找到没有配对上的检测框
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

    // 3.3.1，update matched trackers with assigned detections.
    // each prediction is corresponding to a tracker
    int detIdx, trkIdx;
    for (unsigned int i = 0; i < matchedPairs.size(); i++)
    {
        trkIdx = matchedPairs[i].x;
        detIdx = matchedPairs[i].y;
        trackers[trkIdx].update(detFrameData[detIdx]);
    }

    // 3.3.2，create and initialise new trackers for unmatched detections
    for (auto umd : unmatchedDetections)
    {
        // 创建新tracker时不会调用KalmanTracker的update函数
        KalmanTracker tracker = KalmanTracker(detFrameData[umd]);
        trackers.push_back(tracker);
    }
    
    // 3.3.3，更新未匹配上的跟踪序列
    // 由于之前已经单独调用过predict函数，此处直接用预测的结果进行tracker的跟踪
    // 在predict和update函数中都对latestRect的值进行了更新，此处不再单独操作

    // 3.3.4，remove dead tracklet
    for (auto it = trackers.begin(); it != trackers.end();)
    {
        // 移除情况1：稳定的tracker，连续丢失次数超过阈值max_lost_time
        // 移除情况2：才刚创建的tracker，就连续丢失超过阈值lower_max_lost_time
        if ((it->m_time_since_update > max_lost_time) || 
            (it->m_age == max_lost_time && it->m_time_since_update==lower_max_lost_time))
                it = trackers.erase(it);
        else{
            ++it;
        }
    }

    cycle_time = (double)(getTickCount() - start_time);
    total_time += cycle_time / getTickFrequency();
}


vector<TrackingBox> TRACKER::getReport(){
    // get trackers' output
    frameTrackingResult.clear();
    for (auto it = trackers.begin(); it != trackers.end(); ++it)
    {
        // min_hits不设置为0是因为第一次检测到的目标不用跟踪，不能设大，一般就是1，表示如果连续两帧都检测到目标
        // int time_window = 1; // 表示连续预测的次数
        // if ((it->m_time_since_update < time_window) && it->m_hit_streak >= min_hits)
        if (it->m_observed_num >= min_hits)
        {
            TrackingBox res;
            res.box = it->latestRect;  // 如果有观测值则使用观测值;如果没有观测值就使用预测值
            res.track_id = it->m_id + 1; // +1 as MOT benchmark requires positive
            res.frame_id = frame_count;
            res.obj_conf = it->obj_conf;
            res.class_id = it->class_id;
            frameTrackingResult.push_back(res);
        }
        else{
            // 
        }
        
    }
    return frameTrackingResult;
}