#ifndef EDAC40PANEL_H
#define EDAC40PANEL_H

#include <QDialog>
//#include <winsock.h>
#include "edac40.h"
#include "edacgenerator.h"

#define EDAC40_MAXN 10

namespace Ui {
    class EDAC40Panel;
}

class EDAC40Panel : public QDialog
{
    Q_OBJECT

public:
    explicit EDAC40Panel(QWidget *parent = 0);
    ~EDAC40Panel();

private:
    Ui::EDAC40Panel *ui;
    EDACGenerator e40gen;
    edac40_list_node edac40_list[EDAC40_MAXN];
    int device_num;
    SOCKET edac40_socket;
    unsigned int edac40MaxValue;
    char *e40buf;
    int e40buf_len;
    double C,M,Offset;
    QString VoutString(unsigned code);
public slots:
    void listDevices();
    void chooseDevice(int);
    void updateVoltageValue();
    void commitChanges();
    void stopGenerator();
    void setAdjustmentsEnabled(bool);
    void calculateRange();
    void setDefaultAdjustments();
    void sendAdjustments();
    void setModeRadioButtonsEnabled(bool);
    void saveDefaults();

};

#endif // EDAC40PANEL_H
