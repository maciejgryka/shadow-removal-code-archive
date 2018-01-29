/********************************************************************************
** Form generated from reading UI file 'Scribbler.ui'
**
** Created: Wed 21. Nov 14:43:25 2012
**      by: Qt User Interface Compiler version 4.8.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_SCRIBBLER_H
#define UI_SCRIBBLER_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QHeaderView>
#include <QtGui/QMainWindow>
#include <QtGui/QMenuBar>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_ScribblerClass
{
public:
    QWidget *centralWidget;
    QMenuBar *menuBar;

    void setupUi(QMainWindow *ScribblerClass)
    {
        if (ScribblerClass->objectName().isEmpty())
            ScribblerClass->setObjectName(QString::fromUtf8("ScribblerClass"));
        ScribblerClass->resize(600, 400);
        centralWidget = new QWidget(ScribblerClass);
        centralWidget->setObjectName(QString::fromUtf8("centralWidget"));
        ScribblerClass->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(ScribblerClass);
        menuBar->setObjectName(QString::fromUtf8("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 600, 21));
        ScribblerClass->setMenuBar(menuBar);

        retranslateUi(ScribblerClass);

        QMetaObject::connectSlotsByName(ScribblerClass);
    } // setupUi

    void retranslateUi(QMainWindow *ScribblerClass)
    {
        ScribblerClass->setWindowTitle(QApplication::translate("ScribblerClass", "Scribbler", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class ScribblerClass: public Ui_ScribblerClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_SCRIBBLER_H
