#include <QtGui/QApplication>
#include "edac40panel.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    EDAC40Panel w;
    w.show();

    return a.exec();
}
