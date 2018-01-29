#include "Scribbler.h"
#include "QFileDialog"
#include "QDir"

Scribbler::Scribbler(
    QString input_path,
    QString output_path,
    QWidget *parent,
    Qt::WFlags flags)

	: QMainWindow(parent, flags)
{
	ui.setupUi(this);
	createActions();
	createMenus();
	
	iv = new ImgView(this);
	setCentralWidget(iv);

  if (input_path.isEmpty()) {
    input_path = DEFAULT_IMG;
  }
  iv->setBgImage(input_path, output_path);

	showMaximized();
}

Scribbler::~Scribbler() {
	delete iv;
}

void Scribbler::createActions() {
	// populate File menu
	openAct = new QAction(tr("&Open..."), this);
	connect(openAct, SIGNAL(triggered()), this, SLOT(open()));
  openAct->setShortcut(tr("Ctrl+O"));

	exitAct = new QAction(tr("E&xit"), this);
	exitAct->setShortcut(Qt::Key_Escape);
	connect(exitAct, SIGNAL(triggered()), this, SLOT(close()));

	// edit menu
	undoAct = new QAction(tr("Undo"), this);
	undoAct->setShortcut(tr("Ctrl+Z"));
	connect(undoAct, SIGNAL(triggered()), this, SLOT(undo()));

	// tool menu
	toolBrushAct = new QAction(tr("&Brush"), this);
	toolBrushAct->setShortcut(tr("B"));
	connect(toolBrushAct, SIGNAL(triggered()), this, SLOT(setToolBrush()));

	toolOutlineAct = new QAction(tr("O&utline"), this);
	connect(toolOutlineAct, SIGNAL(triggered()), this, SLOT(setToolOutline()));

	// Mask menu
	saveMaskAct = new QAction(tr("&Save..."), this);
  saveMaskAct->setShortcut(tr("Ctrl+S"));
	connect(saveMaskAct, SIGNAL(triggered()), this, SLOT(saveMask()));

	loadMaskAct = new QAction(tr("&Load..."), this);
	connect(loadMaskAct, SIGNAL(triggered()), this, SLOT(loadMask()));

	clearMaskAct = new QAction(tr("&Clear"), this);
	connect(clearMaskAct, SIGNAL(triggered()), this, SLOT(clearMask()));
}

void Scribbler::createMenus() {
	QMenu *fileMenu = menuBar()->addMenu(tr("&File"));
	fileMenu->addAction(openAct);
	fileMenu->addAction(exitAct);

	QMenu *editMenu = menuBar()->addMenu(tr("&Edit"));
	editMenu->addAction(undoAct);

	QMenu *maskMenu = menuBar()->addMenu(tr("&Mask"));
	maskMenu->addAction(saveMaskAct);
	maskMenu->addAction(loadMaskAct);
	maskMenu->addAction(clearMaskAct);

	QMenu *toolMenu = menuBar()->addMenu(tr("&Tools"));
	toolMenu->addAction(toolBrushAct);
	toolMenu->addAction(toolOutlineAct);
}

void Scribbler::resizeEvent(QResizeEvent *event) {
	iv->setWindowSize(event->size());
}

void Scribbler::setTool(ImgView::Tool newTool) {
	iv->setTool(newTool);
}

void Scribbler::open() {
	iv->setBgImage(QFileDialog::getOpenFileName(
      this,
      tr("Open File"),
      "C:/Work/research/shadow_removal/experiments/2012-03-06/training_images",
      tr("Images (*.jpg *.png *.gif);;All Files(*.*)")));
}

void Scribbler::undo() {
	iv->undo();
}

void Scribbler::setToolOutline() {
	setTool(ImgView::TOOL_OUTLINE);
}

void Scribbler::setToolBrush() {
	setTool(ImgView::TOOL_BRUSH);
}

void Scribbler::saveMask() {
	//iv->saveMask(QFileDialog::getSaveFileName(this, tr("Save File"), "C:/Work/research/shadow_removal/experiments/2012-03-06/training_images", tr("Data files (*.png);;All Files(*.*)")));
  iv->saveMask("");
}

void Scribbler::loadMask() {
	//iv->loadScribbles(QFileDialog::getOpenFileName(this, tr("Open File"), QDir::currentPath() + "\\..\\img\\mask.dat", tr("Data files (*.dat);;All Files(*.*)")));
}

void Scribbler::clearMask() {
	iv->clearMask();
}