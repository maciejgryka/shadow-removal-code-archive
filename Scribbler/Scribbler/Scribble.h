#ifndef SCRIBBLE_H
#define SCRIBBLE_H

#include "ScribblePoint.h"

#include <vector>

#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>

class Scribble {

public:
	Scribble() {};

	Scribble(int type) {
		this->type = type;
		startTime = 0;
	};

	Scribble(int type, QVector<ScribblePoint> data) {
		this->type = type;
		this->data = data;
		startTime = 0;
	};

	~Scribble() {};
	void addPoint(ScribblePoint s) {
		if (startTime == 0) {
			startTime = s.time;
		}
		s.time -= startTime;
		data << s;
	};

	ScribblePoint firstPoint() {
		return data[0];
	};

	ScribblePoint lastPoint() {
		return data[data.size()-1];
	};

	std::vector<std::vector<cv::Point>> getContours(){
		std::vector<cv::Point> pointCvData;
		for (int p = 0; p < data.size(); p++) {
			pointCvData.push_back(data[p].getPointCV());
		}
		
		std::vector<std::vector<cv::Point>> contours;
		contours.push_back(pointCvData);
		return contours;
	};
	
	int type;	// scribble type - see ImgView::Tool enum for possible types
	QVector<ScribblePoint> data;
	qint64 startTime;

	friend QDataStream &operator<<(QDataStream &out, const Scribble &scribble) {
		out << scribble.type << scribble.startTime << scribble.data.size();

		for (int sp = 0; sp < scribble.data.size(); sp++) {
			out << scribble.data[sp].pos[0] << scribble.data[sp].pos[1] << scribble.data[sp].time;
		}
		return out;
	};

	friend QDataStream &operator>>(QDataStream &in, Scribble &scribble){
		int type;
		qint64 startTime;
		int nPoints;
		QVector<ScribblePoint> data;
		in >> type >> startTime >> nPoints;

		for (int sp = 0; sp < nPoints; sp++) {
			ScribblePoint sPoint;
			in >> sPoint.pos[0] >> sPoint.pos[1] >> sPoint.time;
			data << sPoint;
		}
		scribble = Scribble(type, data);
		return in;
	};
};

#endif