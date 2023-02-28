#include "opencv2/opencv.hpp"
#include "track.hpp"
#include "utils.h"

int main(int argc, char **argv)
{
	cout << "+++++++++++++++++ enter main +++++++++++++++++" << endl;
	// 逐帧读取检测结果跑跟踪，检测结果是所有帧保存在一个TXT文件中，每行一个bbox
	string imgfolder = "../data/TUD-Stadtmitte/img1/";
	string detfile = "../data/TUD-Stadtmitte/gt/gt.txt"; // 直接使用真值进行测试

	vector<cv::String> imgpaths;
	getFilePaths(imgfolder, imgpaths, "jpg"); // 获取所有文件路径

	// 从TXT文件中读取检测结果
	std::map<int, vector<Bbox>> det_results; // key是frame_id
	getDetectResults(detfile, det_results);

	TRACK tracker;	  // 创建全局跟踪器
	int frame_id = 1; // frame_id是从1开始编号的
	// 逐帧遍历
	for (auto imgpath : imgpaths)
	{
		cout << "track:" << imgpath << endl;
		// 提取检测结果，将检测框封装成跟踪所需要的数据格式。可在TrackingBox类中增加其他属性。
		vector<TrackingBox> detFrameData;
		vector<Bbox> bboxes = det_results[frame_id];
		for (int i = 0; i < bboxes.size(); ++i)
		{
			TrackingBox cur_box;
			cur_box.box = bboxes[i].rect;
			cur_box.track_id = -1;  // 初始化为-1
			cur_box.frame_id = frame_id;
			detFrameData.push_back(cur_box);
		}
		++frame_id;
		// 跟踪
		vector<TrackingBox> tracking_results = tracker.update(detFrameData);
		// 可视化跟踪结果
		Mat frame = cv::imread(imgpath);
		for (TrackingBox it : tracking_results)
		{
			cv::Rect object(it.box.x, it.box.y, it.box.width, it.box.height);
			cv::rectangle(frame, object, tracker.randColor[it.track_id % 255], 2);
			cv::putText(frame,
						to_string(it.track_id),
						cv::Point2f(it.box.x, it.box.y),
						cv::FONT_HERSHEY_DUPLEX,
						1,
						tracker.randColor[it.track_id % 255]);
		}
		cv::imshow(imgpath, frame);
		cv::waitKey(0);
	}

	return 0;
}