#include <QString>
#include <QtGui/QApplication>

#include "Scribbler.h"

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
  QString input("");
  QString output("");
  if (argc > 1) {
    input = QString(argv[1]);
  }
  if (argc > 2) {
    output = QString(argv[2]);
  }
	Scribbler w(input, output);
	w.show();
	return a.exec();
}
