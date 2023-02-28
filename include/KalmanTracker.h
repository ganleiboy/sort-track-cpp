///////////////////////////////////////////////////////////////////////////////
// KalmanTracker.h: KalmanTracker Class Declaration

#ifndef KALMAN_H
#define KALMAN_H 2

#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

#define StateType Rect_<float>

// This class represents the internel state of individual tracked objects observed as bounding box.
class KalmanTracker
{
public:
	KalmanTracker()
	{
		init_kf(StateType());
		m_time_since_update = 0;
		m_hits = 0;
		m_hit_streak = 0;
		m_age = 0;
		m_id = kf_count;
		kf_count++;
	}
	KalmanTracker(StateType initRect)
	{
		init_kf(initRect);
		m_time_since_update = 0;
		m_hits = 0;
		m_hit_streak = 0;
		m_age = 0;
		m_id = kf_count;
		kf_count++;
	}

	~KalmanTracker()
	{
		m_history.clear();
	}

	StateType predict();
	void update(StateType stateMat);

	StateType get_state();
	StateType get_rect_xysr(float cx, float cy, float s, float r);

	StateType lastRect;
	static int kf_count;  // 每次调用构造函数kf_count就会加1

	int m_time_since_update;
	int m_hits;		  // 该目标框进行更新的总次数。每执行update一次，便hits+=1
	int m_hit_streak; // 连续更新的次数。判断当前是否做了更新，大于等于1的说明做了更新，只要连续帧中没有做连续更新，hit_streak就会清零
	int m_age;		  // 该目标框进行预测的总次数。每执行predict一次，便age+=1
	int m_id;		  // track_id。每次调用构造函数kf_count就会加1

private:
	void init_kf(StateType stateMat);

	cv::KalmanFilter kf;
	cv::Mat measurement;  // 观测值

	std::vector<StateType> m_history;  // 保存单个目标框连续预测的多个结果到history列表中，一旦执行update就会清空
};

#endif