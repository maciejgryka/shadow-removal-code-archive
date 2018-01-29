#include <vector>

#include "ImgView.h"
#include <QColor>
#include <QFile>
#include <QDateTime>
#include "qimage.h"

using namespace cv;

ImgView::ImgView(QWidget *parent): 
	QGraphicsView(parent),
	brushColor(255, 255, 255, 64)
{
	scene = new QGraphicsScene(this);
	setScene(scene);

	setTool(TOOL_BRUSH);
}

ImgView::~ImgView() {
	delete scene;
	for (int s = 0; s < scribbles.size(); s++) {
		delete scribbles[s];
	}
}

void ImgView::undo() {
	if (masks.size() > 1) {
		masks.pop();
		updateBgImage();
	}
}

QString ImgView::getMaskPath(const QString& shadPath) {
  QStringList filePathParts(shadPath.split('\\'));
  QString fileName(filePathParts.last());

  QString directory("");
  for (int i(0); i < filePathParts.size() - 1; ++i) {
    directory += filePathParts[i] + "\\";
  }

  QString extension(fileName.split('.')[1]);
  QStringList name(fileName.split('.')[0].split('_'));
  QString maskPath("");
  for (int i(0); i < name.size() - 1; ++i) {
    maskPath += name[i] + QString("_");
  }
  maskPath += QString(MASK_SUFFIX) + QString(".") + extension;
  return directory + maskPath;
}

void ImgView::setBgImage(const QString& input_path, const QString& output_path) {
  if (output_path.isEmpty()) {
    maskPath = getMaskPath(input_path).toStdString();
  } else {
    maskPath = output_path.toStdString();
  }
	bgImage = QPixmap(input_path);
	masks.push(Mat::zeros(bgImage.height(), bgImage.width(), CV_8UC3));
	
	scaleToImage();

	scene->clear();
	updateBgImage();
	updateCursor();
}

void ImgView::resetScene() {
	scene->clear();
	updateBgImage();
}

void ImgView::updateCursor() {
	if (currentTool == TOOL_BRUSH) {
		int s;
		if (this->windowSize.width() > 0) {
			s = (pen.width()+1) * scaleFactor;
		} else {
			s = (16+1);
		}
		QPixmap* im = new QPixmap(s,s);
		im->fill(Qt::transparent);
		QPainter painter(im);
		painter.setPen(Qt::NoPen);
		painter.setBrush(brushColor);
		painter.drawEllipse(0,0,s,s);
		mCursor = QCursor(*im);
	} else if (currentTool == TOOL_OUTLINE) {
		mCursor = Qt::ArrowCursor;
	}

	setCursor(mCursor);
}

void ImgView::setWindowSize(QSize size) {
	windowSize = size;
	scaleToImage();
	updateCursor();
}

void ImgView::setTool(Tool newTool) {
	currentTool = newTool;
	setPen(newTool);
}

void ImgView::saveMask(QString path) {
	//// save recorded scribbles
	//saveScribbles(path);
	
	// save the image
	imwrite(maskPath, masks.top());
}

void ImgView::saveScribbles(QString path) {
	QFile outFile(path);
	outFile.open(QIODevice::WriteOnly);
	QDataStream out(&outFile);
	out << scribbles.size();
	for (int sc = 0; sc < scribbles.size(); sc++) {
		out << *(scribbles[sc]);
	}
	outFile.close();
}

void ImgView::clearMask() {
	masks.top() = Mat::zeros(bgImage.height(), bgImage.width(), CV_8UC3);
	updateBgImage();
	deleteScribbles();
}

void ImgView::deleteScribbles() {
	scribbles.clear();
}

Scribble* ImgView::lastScribble() {
	return scribbles[scribbles.size()-1];
}

void ImgView::mousePressEvent(QMouseEvent *event) {
	lastPos = mapToScene(event->pos());
	scribbles << new Scribble(currentTool);
	//updateScribble(lastPos);

	masks.push(masks.top().clone());

	switch (currentTool) {
	case TOOL_BRUSH:
		//updateScribble(lastPos);
		drawCircle(QPoint(lastPos.x(), lastPos.y()));
		break;
	case TOOL_OUTLINE:
		updateScribble(lastPos);
		drawCircleScene(QPoint(lastPos.x(), lastPos.y()));
		break;
	}
}

void ImgView::mouseMoveEvent(QMouseEvent *event) {
	QPointF p = mapToScene(event->pos());
	
	switch (currentTool) {
	case TOOL_BRUSH:
		//updateScribble(p);
		drawCircle(p);
		break;
	case TOOL_OUTLINE:
		updateScribble(p);
		drawCircleScene(p);
		break;
	}
}

void ImgView::mouseReleaseEvent(QMouseEvent *event) {
	QPointF p = mapToScene(event->pos());

	switch (currentTool) {
	case TOOL_OUTLINE:
		drawLine(lastScribble()->lastPoint().getQPointF(), lastScribble()->firstPoint().getQPointF());
		break;
	}
	lastPos = p;
}

void ImgView::wheelEvent(QWheelEvent* event) {
	if (currentTool != TOOL_BRUSH) {
		return;
	}
	int d = event->delta() / 64;
	
	pen.setWidth(pen.width() + d);

	int lowThresh(2);
	int highThresh(bgImage.width());

	if (pen.width() < lowThresh) {
		pen.setWidth(lowThresh);
	} else if (pen.width() > highThresh) {
		pen.setWidth(highThresh);
	}
	updateCursor();

}

void ImgView::keyPressEvent(QKeyEvent *event) {
	switch (event->key()) {
	case Qt::Key_BracketLeft:
		if (pen.width() > 1) {
			pen.setWidth(pen.width() - 1);
			updateCursor();
		}
		break;
	case Qt::Key_BracketRight:
		pen.setWidth(pen.width() + 1);
		updateCursor();
	}
}

void ImgView::setPen(Tool t) {
	int brushWidth = 5;
	int outlineWidth = 1;
	QColor colorWhite(255, 255, 255, 10);

	switch (t) {
	case TOOL_BRUSH:
		pen.setWidth(brushWidth);
		pen.setColor(colorWhite);
		updateCursor();
		break;
	case TOOL_OUTLINE:
		pen.setWidth(outlineWidth);
		pen.setColor(colorWhite);
		updateCursor();
		break;
	}
}

void ImgView::scaleToImage() {
	if (windowSize.height() > windowSize.width()) {
		scaleFactor = float(windowSize.width()-50)/bgImage.size().width();
	} else {
		scaleFactor = float(windowSize.height()-50)/bgImage.size().height();
	}
	resetTransform();
	scale(scaleFactor, scaleFactor);
}

// adds p to the current scribble
void ImgView::updateScribble(QPointF p) {
	lastScribble()->addPoint(ScribblePoint(p.x(), p.y(), QDateTime::currentMSecsSinceEpoch()));
}

void ImgView::updateBgImage() {
	cv::Mat cvbg(qimage2mat(bgImage.toImage()));
	
	cv::Mat res = cvbg + masks.top()*0.2;
	
  res.convertTo(res, CV_8U);
  cvtColor(res, res, CV_BGR2RGB);
  qMask = QImage(res.data, res.cols, res.rows, res.step, QImage::Format_RGB888);

	scene->clear();
	scene->addPixmap(QPixmap::fromImage(qMask));
}

void ImgView::drawCircle(QPointF p) {
	qreal brushSize = pen.widthF();
	//scene->addEllipse(p.x()-brushSize/2, p.y()-brushSize/2, brushSize, brushSize, pen);
	
	RotatedRect r(Point2f(p.x(), p.y()), Size2f(brushSize, brushSize), 0.0);
	ellipse(masks.top(), r, Scalar(255.0, 255.0, 255.0), -1);
	updateBgImage();
}

void ImgView::drawCircleScene(QPointF p) {
	float r(pen.width());
	scene->addEllipse(p.x() - r, p.y() - r, r, r, pen);
}

QGraphicsLineItem* ImgView::drawLine(QPointF p) {
	return drawLine(lastPos, p);
}

QGraphicsLineItem* ImgView::drawLine(QPointF q, QPointF p) {
	return scene->addLine(QLineF(q, p), pen);
}

Mat ImgView::qimage2mat(const QImage& qimage) { 
  cv::Mat mat = cv::Mat(qimage.height(), 
                        qimage.width(), 
                        CV_8UC4, 
                        (uchar*)qimage.bits(), 
                        qimage.bytesPerLine()); 
  cv::Mat mat2 = cv::Mat(mat.rows, mat.cols, CV_8UC3 ); 
  int from_to[] = { 0,0,  1,1,  2,2 }; 
  cv::mixChannels( &mat, 1, &mat2, 1, from_to, 3 ); 
  return mat2; 
}

QImage ImgView::mat2qimage(const Mat& mat) { 
  Mat rgb(mat.clone()); 
  rgb.convertTo(rgb, CV_8U);
  cvtColor(rgb, rgb, CV_BGR2RGB);
  QImage qMask(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
  return qMask;
}