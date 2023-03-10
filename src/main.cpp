#include "opencv2/opencv.hpp"
#include "track.h"
#include "utils.h"

int main(int argc, char **argv)
{
	cout << "+++++++++++++++++ enter main +++++++++++++++++" << endl;
	// 逐帧读取检测结果跑跟踪，检测结果是所有帧保存在一个TXT文件中，每行一个bbox
	string imgfolder = "../data/TUD-Stadtmitte/img1/";
	string detfile = "../data/TUD-Stadtmitte/gt/gt.txt"; // 直接使用真值进行测试
	string savefolder = "../results/";	// 保存可视化结果的目录

	vector<cv::String> imgpaths;
	getFilePaths(imgfolder, imgpaths, "jpg"); // 获取所有文件路径

	// 从TXT文件中读取检测结果
	std::map<int, vector<Bbox>> det_results; // key是frame_id
	getDetectResults(detfile, det_results);

	TRACKER tracker;  // 创建全局跟踪器
	int frame_id = 1; // frame_id是从1开始编号的
	// 逐帧遍历
	for (auto imgpath : imgpaths)
	{
		cout << "frame_id: " << frame_id << ", " << imgpath << endl;
		// 提取检测结果，将检测框封装成跟踪所需要的数据格式。可在TrackingBox类中增加其他属性。
		vector<TrackingBox> detFrameData;
		vector<Bbox> bboxes = det_results[frame_id];
		for (int i = 0; i < bboxes.size(); ++i)
		{
			TrackingBox cur_box(bboxes[i]);
			cur_box.frame_id = frame_id;
			detFrameData.push_back(cur_box);
		}
		frame_id++;
		// 跟踪 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		tracker.update(detFrameData);
		vector<TrackingBox> tracking_results = tracker.getReport();
		// 可视化跟踪结果
		Mat frame = cv::imread(imgpath);
		string imgname = imgpath.substr(imgpath.find_last_of('/') + 1);  // 提取文件名
		drawPic(frame, savefolder + imgname, tracking_results, tracker);
	}

	return 0;
}