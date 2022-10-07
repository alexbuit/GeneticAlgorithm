#include "edac40panel.h"
#include "ui_edac40panel.h"
#include "edac40.h"
#include "edacgenerator.h"

#define EDAC40_DISCOVER_TIMEOUT 750 // milliseconds
#define EDAC40_DISCOVER_ATTEMPTS 2

EDAC40Panel::EDAC40Panel(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::EDAC40Panel)
{
    int i;
    edac40_channel_value e40data[40];
    edac40_init();
    edac40_socket=-1;
    for(i=0; i<40; i++)
      {
        e40data[i].channel=i;
        e40data[i].value=0;
      }
    e40buf_len=edac40_prepare_packet(e40data,40,&e40buf);
    ui->setupUi(this);
    listDevices();
    //setDefaultAdjustments();
}

EDAC40Panel::~EDAC40Panel()
{
    edac40_finish();
    delete ui;
}

void EDAC40Panel::listDevices()
{
  //int i;
  //for(i=0; i<ui->deviceComboBox->count();i++) ui->deviceComboBox->removeItem(i);
  ui->deviceComboBox->clear();
  if((device_num=edac40_list_devices(edac40_list, EDAC40_MAXN, EDAC40_DISCOVER_TIMEOUT, EDAC40_DISCOVER_ATTEMPTS))>0)
    {
      for(int i=0; i<device_num; i++) ui->deviceComboBox->addItem(edac40_list[i].MACAddress);
      ui->commitButton->setEnabled(true);
      ui->stopButton->setEnabled(false);
      ui->adjustmentsEnableCheckBox->setEnabled(true);
    }
  else
    {
      ui->deviceComboBox->addItem("<no device found>");
      ui->commitButton->setEnabled(false);
      ui->stopButton->setEnabled(false);
      ui->adjustmentsEnableCheckBox->setEnabled(false);
    }
  ui->deviceComboBox->setCurrentIndex(0);
  chooseDevice(0);
}

void EDAC40Panel::chooseDevice(int deviceIndex)
{
  if(edac40_socket>0)
    {
      edac40_close(edac40_socket);
      edac40_socket=-1;
    }
  if(device_num>0)
    {
      ui->ipLabel->setText(edac40_list[deviceIndex].IPAddress);
      edac40_socket=edac40_open(edac40_list[deviceIndex].IPAddress,0);
      calculateRange();
    }
  else ui->ipLabel->setText("XXX.XXX.XXX.XXX");
}

void EDAC40Panel::updateVoltageValue()
{
    if(ui->valueUnitButton->isChecked()) edac40MaxValue=0xFFFF;
    if(ui->valueThreeQuartersButton->isChecked()) edac40MaxValue=3*(0xFFFF/4);
    if(ui->valueOneHalfButton->isChecked()) edac40MaxValue=0xFFFF/2;
    if(ui->valueOneQuarterButton->isChecked()) edac40MaxValue=0xFFFF/4;
    if(ui->valueZeroButton->isChecked()) edac40MaxValue=0;
    if(ui->valueOtherButton->isChecked()) edac40MaxValue=ui->valueSpinBox->value();
    calculateRange();
}

void EDAC40Panel::commitChanges()
{

  int mode;
  updateVoltageValue();
  if(ui->constantButton->isChecked()) mode=0;
  if(ui->squareButton->isChecked()) mode=1;
  if(ui->rampButton->isChecked()) mode=2;
  e40gen.setParameters(edac40_socket, mode, edac40MaxValue);
  ui->commitButton->setEnabled(false);
  ui->stopButton->setEnabled(true);
  setModeRadioButtonsEnabled(false);
  setAdjustmentsEnabled(false);
  ui->adjustmentsEnableCheckBox->setEnabled(false);
  ui->deviceComboBox->setEnabled(false);
  ui->refreshButton->setEnabled(false);
  e40gen.start();
}

void EDAC40Panel::stopGenerator()
{
  e40gen.stop();
  ui->commitButton->setEnabled(true);
  ui->stopButton->setEnabled(false);
  setModeRadioButtonsEnabled(true);
  ui->adjustmentsEnableCheckBox->setEnabled(true);
  ui->deviceComboBox->setEnabled(true);
  ui->refreshButton->setEnabled(true);
  setAdjustmentsEnabled(ui->adjustmentsEnableCheckBox->isChecked());
}

void EDAC40Panel::setAdjustmentsEnabled(bool enabled)
{
   ui->offsetDACSlider->setEnabled(enabled);
   ui->offsetDACSpinBox->setEnabled(enabled);
   ui->offsetDACLabel->setEnabled(enabled);

   ui->gainSlider->setEnabled(enabled);
   ui->gainSpinBox->setEnabled(enabled);
   ui->gainLabel->setEnabled(enabled);

   //ui->offsetSlider->setEnabled(enabled);
   //ui->offsetSpinBox->setEnabled(enabled);
   //ui->offsetLabel->setEnabled(enabled);

   ui->estimatedRangeLabel->setEnabled(enabled);
   ui->rangeMinLabel->setEnabled(enabled);
   ui->dotsLabel->setEnabled(enabled);
   ui->rangeMaxLabel->setEnabled(enabled);

   ui->resetDefaultsButton->setEnabled(enabled);

   ui->saveDefaultsButton->setEnabled(enabled);
}


void EDAC40Panel::sendAdjustments()
{
    // send packages to the DAC unit to set gains, offsets, offset DAC (global offest)
    char *buffer;
    int buf_len;

    unsigned int offset=ui->offsetDACSpinBox->value();
    edac40_set(edac40_socket,EDAC40_SET_OFFSET_DACS,1,offset/* & 0x3FFF*/);

    // It was decided not to use individual DACs offest adjustment to avoid confusion
    // It is still used by the unit itself and can be changed from user software,
    // but here they are just set to default values and later stored to NVRAM
    buf_len=edac40_prepare_packet_fill(0x8000,EDAC40_SET_OFFSET,&buffer);
    edac40_send_packet(edac40_socket,buffer,buf_len);
    free(buffer);

    unsigned int gain=ui->gainSpinBox->value();
    buf_len=edac40_prepare_packet_fill(gain,EDAC40_SET_GAIN,&buffer);
    edac40_send_packet(edac40_socket,buffer,buf_len);
    free(buffer);

}

void EDAC40Panel::calculateRange()
{
  C=0x8000; // Offset, default value is used
  M=ui->gainSpinBox->value();
  Offset=ui->offsetDACSpinBox->value();
  ui->rangeMinLabel->setText(VoutString(0));
  ui->rangeMaxLabel->setText(VoutString(0xFFFF));
}

QString EDAC40Panel::VoutString(unsigned code)
{
  double V;
  double dacCode;
  char formatString[256];
  QString  resultString;

  dacCode=code*(M+1)/65535.0+C-32768.0;
  if(dacCode>65535) dacCode=65535;
  if(dacCode<0) dacCode=0;
  V=12*(dacCode-4*Offset)/65535.0;
  strcpy(formatString,"%-7.3lf");
  if(V>=12.001)
    {
      V=12;
      strcpy(formatString,"<font color=red>%-7.3lf</font>");
    }
  if(V<=-12.001)
    {
      V=-12;
      strcpy(formatString,"<font color=blue>%-7.3lf</font>");
    }
  resultString.sprintf(formatString,V);
  return resultString;
}

void EDAC40Panel::setDefaultAdjustments()
{
  // first touch controls, otherwise if present values
  // are equal to the defaults, signals wouldn't be emitted
  ui->gainSpinBox->setValue(0xFFFF);
  ui->gainSpinBox->setValue(0x10000);

  ui->offsetDACSpinBox->setValue(0x1FFF);
  ui->offsetDACSpinBox->setValue(0x2000);

  calculateRange();

}

void EDAC40Panel::setModeRadioButtonsEnabled(bool enable)
{
  ui->valueUnitButton->setEnabled(enable);
  ui->valueThreeQuartersButton->setEnabled(enable);
  ui->valueOneHalfButton->setEnabled(enable);
  ui->valueOneQuarterButton->setEnabled(enable);
  ui->valueZeroButton->setEnabled(enable);
  ui->valueOtherButton->setEnabled(enable);
  ui->constantButton->setEnabled(enable);
  ui->squareButton->setEnabled(enable);
  ui->rampButton->setEnabled(enable);
}

void EDAC40Panel::saveDefaults()
{
    char *e40buf;
    unsigned buf_len;
    buf_len=edac40_prepare_packet_fill(32768,EDAC40_SET_OFFSET,&e40buf);
    edac40_send_packet(edac40_socket, e40buf, buf_len);
    free(e40buf);
    // instruct the module to store actual settings to NVRAM
    edac40_save_defaults(edac40_socket);
}
