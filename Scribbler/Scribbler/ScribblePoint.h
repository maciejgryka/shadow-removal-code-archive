#ifndef SCRIBBLEPOINT_H
#define SCRIBBLEPOINT_H

#include <qpoint.h>
#include <opencv2\opencv.hpp>

// Storage class with XY postion of a point and a timestamp
class ScribblePoint {

public:
	ScribblePoint() {};

	ScribblePoint(qreal pos[2], qint64 time) {
		ScribblePoint(pos[0], pos[1], time);
	};

	ScribblePoint(qreal x, qreal y, qint64 time) {
		pos[0] = x;
		pos[1] = y;
		this->time = time;
	}

	~ScribblePoint() {};

	QPointF getQPointF() {
		return QPointF(pos[0], pos[1]);
	};
	
	cv::Point getPointCV() {
		return cv::Point(pos[0], pos[1]);
	};

	qint64 time;
	qreal pos[2];

private:

	friend QDataStream &operator<<(QDataStream &out, const ScribblePoint &point) {
		out << point.pos[0] << point.pos[1] << point.time;
		return out;
	};

	friend QDataStream &operator>>(QDataStream &in, ScribblePoint &point){
		qreal pos[2];
		qint64 time;
		in >> pos[0] >> pos[1] >> time;
		point = ScribblePoint(pos, time);
		return in;
	};

};

#endif