///////////////////////////////////////////////////////////////////////////////
// KalmanTracker.cpp: KalmanTracker Class Implementation Declaration

#include "KalmanTracker.h"

int KalmanTracker::kf_count = 0;

// initialize Kalman filter
void KalmanTracker::init_kf(StateType stateMat)
{
	int stateNum = 7;  // 一个7维的状态更新向量：[u,v,s,r,u^,v^,s^]T。Note：u^,v^,s^表示运动速度
	int measureNum = 4;  // 一个4维的观测输入，即中心面积的形式[x,y,s,r]，即[检测框中心x坐标,y坐标,面积,宽高比]。
	kf = KalmanFilter(stateNum, measureNum, 0);

	// 状态转移矩阵(A)。默认两帧的时间间隔是1，无量纲
	// Note：如果要计算实际运动速度，则右上角的三个1需要更改为两帧的时间间隔dt，并在每次调用predict()函数之前进行重置
	kf.transitionMatrix = (Mat_<float>(stateNum, stateNum) << 
						   1, 0, 0, 0, 1, 0, 0,
						   0, 1, 0, 0, 0, 1, 0,
						   0, 0, 1, 0, 0, 0, 1,
						   0, 0, 0, 1, 0, 0, 0,
						   0, 0, 0, 0, 1, 0, 0,
						   0, 0, 0, 0, 0, 1, 0,
						   0, 0, 0, 0, 0, 0, 1);

	measurement = Mat::zeros(measureNum, 1, CV_32F);		// 观测值，初始化为0
	
	setIdentity(kf.measurementMatrix);						// 测量矩阵 H
	setIdentity(kf.processNoiseCov, Scalar::all(1e-2));		// 系统误差 Q
	setIdentity(kf.measurementNoiseCov, Scalar::all(1e-1)); // 测量误差 R
	setIdentity(kf.errorCovPost, Scalar::all(1));			// 最小均方误差 P'(k))

	// initialize state vector with bounding box in [cx,cy,s,r] style
	kf.statePost.at<float>(0, 0) = stateMat.x + stateMat.width / 2;	 // 检测框中心x坐标
	kf.statePost.at<float>(1, 0) = stateMat.y + stateMat.height / 2; // 检测框中心y坐标
	kf.statePost.at<float>(2, 0) = stateMat.area();					 // 检测框面积
	kf.statePost.at<float>(3, 0) = stateMat.width / stateMat.height; // 检测框宽高比
}

// Predict the estimated bounding box.
StateType KalmanTracker::predict()
{
	// predict
	Mat p = kf.predict();  // 计算预测的状态值，一个7维的状态更新向量，最后三个元素是运动速度
	m_age += 1;
	
	if (m_time_since_update > m_max_missing_observed_num)
		m_hit_streak = 0;
	m_time_since_update += 1;

	StateType predictBox = get_rect_xysr(p.at<float>(0, 0), p.at<float>(1, 0), p.at<float>(2, 0), p.at<float>(3, 0));

	m_history.push_back(predictBox);
	return m_history.back();  // 返回对vector最后一个元素的引用
}

// Update the state vector with observed bounding box.
void KalmanTracker::update(TrackingBox track_box)
{
	this->lastRect = track_box.box;

	m_time_since_update = 0;  // 每次观察到目标就重置为0
	m_history.clear();
	m_observed_num += 1;
	m_hit_streak += 1;

	obj_conf = track_box.obj_conf;
	class_id = track_box.class_id;

	// measurement
	measurement.at<float>(0, 0) = track_box.box.x + track_box.box.width / 2;
	measurement.at<float>(1, 0) = track_box.box.y + track_box.box.height / 2;
	measurement.at<float>(2, 0) = track_box.box.area();
	measurement.at<float>(3, 0) = track_box.box.width / track_box.box.height;

	// update
	kf.correct(measurement);  // 根据测量值更新状态值
}

// Return the current state vector
StateType KalmanTracker::get_state()
{
	Mat s = kf.statePost;
	return get_rect_xysr(s.at<float>(0, 0), s.at<float>(1, 0), s.at<float>(2, 0), s.at<float>(3, 0));
}

// Convert bounding box from [cx,cy,s,r] to [x,y,w,h] style.
StateType KalmanTracker::get_rect_xysr(float cx, float cy, float s, float r)
{
	float w = sqrt(s * r);
	float h = s / w;
	float x = (cx - w / 2);
	float y = (cy - h / 2);

	if (x < 0 && cx > 0)
		x = 0;
	if (y < 0 && cy > 0)
		y = 0;

	return StateType(x, y, w, h);
}

/*
// --------------------------------------------------------------------
// Kalman Filter Demonstrating, a 2-d ball demo
// --------------------------------------------------------------------
const int winHeight = 600;
const int winWidth = 800;
Point mousePosition = Point(winWidth >> 1, winHeight >> 1);
// mouse event callback
void mouseEvent(int event, int x, int y, int flags, void *param)
{
	if (event == CV_EVENT_MOUSEMOVE) {
		mousePosition = Point(x, y);
	}
}
void TestKF();
void main()
{
	TestKF();
}
void TestKF()
{
	int stateNum = 4;
	int measureNum = 2;
	KalmanFilter kf = KalmanFilter(stateNum, measureNum, 0);
	// initialization
	Mat processNoise(stateNum, 1, CV_32F);
	Mat measurement = Mat::zeros(measureNum, 1, CV_32F);
	kf.transitionMatrix = *(Mat_<float>(stateNum, stateNum) <<
		1, 0, 1, 0,
		0, 1, 0, 1,
		0, 0, 1, 0,
		0, 0, 0, 1);
	setIdentity(kf.measurementMatrix);
	setIdentity(kf.processNoiseCov, Scalar::all(1e-2));
	setIdentity(kf.measurementNoiseCov, Scalar::all(1e-1));
	setIdentity(kf.errorCovPost, Scalar::all(1));
	randn(kf.statePost, Scalar::all(0), Scalar::all(winHeight));
	namedWindow("Kalman");
	setMouseCallback("Kalman", mouseEvent);
	Mat img(winHeight, winWidth, CV_8UC3);
	while (1)
	{
		// predict
		Mat prediction = kf.predict();
		Point predictPt = Point(prediction.at<float>(0, 0), prediction.at<float>(1, 0));
		// generate measurement
		Point statePt = mousePosition;
		measurement.at<float>(0, 0) = statePt.x;
		measurement.at<float>(1, 0) = statePt.y;
		// update
		kf.correct(measurement);
		// visualization
		img.setTo(Scalar(255, 255, 255));
		circle(img, predictPt, 8, CV_RGB(0, 255, 0), -1); // predicted point as green
		circle(img, statePt, 8, CV_RGB(255, 0, 0), -1); // current position as red
		imshow("Kalman", img);
		char code = (char)waitKey(100);
		if (code == 27 || code == 'q' || code == 'Q')
			break;
	}
	destroyWindow("Kalman");
}
*/