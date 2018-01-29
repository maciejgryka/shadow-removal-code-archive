#ifndef IMGVIEW_H
#define IMGVIEW_H

#include <stack>

#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QMouseEvent>
#include <qbitmap.h>

#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>

#include "Scribble.h"


#define MASK_SUFFIX "smask"


class ImgView : public QGraphicsView
{
	Q_OBJECT

public:
	ImgView(QWidget *parent);
	~ImgView();

	enum Tool {
			TOOL_OUTLINE = 0,	// outline of the shadow with a lasso tool
			TOOL_BRUSH,			// broad paintbrush
	};

	void setBgImage(const QString& path, const QString& output_path = "");
	void resetScene();
	void setWindowSize(QSize size);
	void setTool(Tool newTool);
	void saveMask(QString path);
	void saveScribbles(QString path);
	void clearMask();
	void deleteScribbles();
	void undo();
	Scribble* lastScribble();

protected:
	void mousePressEvent(QMouseEvent *event);
	void mouseMoveEvent(QMouseEvent *event);
	void mouseReleaseEvent(QMouseEvent *event);
	void wheelEvent(QWheelEvent* event);

	void keyPressEvent(QKeyEvent *event);

private:
	QGraphicsScene *scene;
	QPixmap bgImage;
  std::string maskPath;

	cv::Mat mask;
	std::stack<cv::Mat> masks;
	QImage qMask;
	
	QSize windowSize;
	float scaleFactor;
	QPen pen;
	QPointF lastPos;

	QCursor mCursor;
	QColor brushColor;

	Tool currentTool;

	QVector<Scribble*> scribbles;

	void scaleToImage();
	void updateScribble(QPointF p);
	void updateBgImage();
	void drawCircle(QPointF p);
	void drawCircleScene(QPointF p);
	QGraphicsLineItem* drawLine(QPointF p);
	QGraphicsLineItem* drawLine(QPointF q, QPointF p);
  QString getMaskPath(const QString& shadPath);

  static cv::Mat qimage2mat(const QImage& qimage) ;
  static QImage mat2qimage(const cv::Mat& mat);

	void updateCursor();

	void setPen(Tool t);
};

#endif // IMGVIEW_H
