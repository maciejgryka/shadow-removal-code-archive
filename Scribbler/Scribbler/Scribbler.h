#ifndef SCRIBBLER_H
#define SCRIBBLER_H

#include <QString>
#include <QtGui/QMainWindow>
#include <QResizeEvent>
#include "ui_Scribbler.h"

#include "ImgView.h"


#define DEFAULT_IMG "C:/Work/research/shadow_removal/experiments/2012-03-06/training_images/001_0043_wood2_shad.png"


class Scribbler : public QMainWindow
{
	Q_OBJECT

public:
	Scribbler(
      QString input_path = "",
      QString output_path = "",
      QWidget *parent = 0,
      Qt::WFlags flags = 0);
	~Scribbler();

protected:
	void createActions();
	void createMenus();
	void resizeEvent(QResizeEvent *event);

private:
	void setTool(ImgView::Tool newTool);

	Ui::ScribblerClass ui;
	ImgView *iv;

	QAction *openAct;
	QAction *exitAct;

	QAction *undoAct;

	QAction *toolOutlineAct;
	QAction *toolBrushAct;
	
	QAction *saveMaskAct;
	QAction *loadMaskAct;
	QAction *clearMaskAct;

private slots:
	void open();
	void undo();
	void saveMask();
	void loadMask();
	void clearMask();

	void setToolOutline();
	void setToolBrush();
};

#endif // SCRIBBLER_H
