/********************************************************************************
** Form generated from reading UI file 'edac40panel.ui'
**
** Created: Wed Dec 4 18:29:51 2013
**      by: Qt User Interface Compiler version 4.8.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_EDAC40PANEL_H
#define UI_EDAC40PANEL_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QCheckBox>
#include <QtGui/QComboBox>
#include <QtGui/QDialog>
#include <QtGui/QGridLayout>
#include <QtGui/QGroupBox>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QPushButton>
#include <QtGui/QRadioButton>
#include <QtGui/QSlider>
#include <QtGui/QSpacerItem>
#include <QtGui/QSpinBox>
#include <QtGui/QVBoxLayout>

QT_BEGIN_NAMESPACE

class Ui_EDAC40Panel
{
public:
    QVBoxLayout *verticalLayout_4;
    QHBoxLayout *horizontalLayout_2;
    QLabel *label;
    QComboBox *deviceComboBox;
    QPushButton *refreshButton;
    QHBoxLayout *horizontalLayout_4;
    QLabel *label_2;
    QLabel *ipLabel;
    QHBoxLayout *horizontalLayout_3;
    QGroupBox *groupBox;
    QVBoxLayout *verticalLayout;
    QRadioButton *valueUnitButton;
    QRadioButton *valueThreeQuartersButton;
    QRadioButton *valueOneHalfButton;
    QRadioButton *valueOneQuarterButton;
    QRadioButton *valueZeroButton;
    QHBoxLayout *horizontalLayout;
    QRadioButton *valueOtherButton;
    QSpinBox *valueSpinBox;
    QGroupBox *groupBox_2;
    QVBoxLayout *verticalLayout_2;
    QRadioButton *constantButton;
    QRadioButton *squareButton;
    QRadioButton *rampButton;
    QSpacerItem *verticalSpacer;
    QVBoxLayout *verticalLayout_3;
    QPushButton *commitButton;
    QPushButton *stopButton;
    QGroupBox *groupBox_3;
    QGridLayout *gridLayout;
    QLabel *estimatedRangeLabel;
    QHBoxLayout *horizontalLayout_5;
    QLabel *rangeMinLabel;
    QLabel *dotsLabel;
    QLabel *rangeMaxLabel;
    QSpacerItem *horizontalSpacer_2;
    QLabel *offsetDACLabel;
    QSpinBox *offsetDACSpinBox;
    QSlider *offsetDACSlider;
    QLabel *gainLabel;
    QSpinBox *gainSpinBox;
    QSlider *gainSlider;
    QCheckBox *adjustmentsEnableCheckBox;
    QPushButton *resetDefaultsButton;
    QPushButton *saveDefaultsButton;

    void setupUi(QDialog *EDAC40Panel)
    {
        if (EDAC40Panel->objectName().isEmpty())
            EDAC40Panel->setObjectName(QString::fromUtf8("EDAC40Panel"));
        EDAC40Panel->resize(388, 445);
        QFont font;
        font.setStyleStrategy(QFont::PreferDefault);
        EDAC40Panel->setFont(font);
        EDAC40Panel->setWindowOpacity(1);
        verticalLayout_4 = new QVBoxLayout(EDAC40Panel);
        verticalLayout_4->setSpacing(6);
        verticalLayout_4->setContentsMargins(11, 11, 11, 11);
        verticalLayout_4->setObjectName(QString::fromUtf8("verticalLayout_4"));
        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setSpacing(6);
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        label = new QLabel(EDAC40Panel);
        label->setObjectName(QString::fromUtf8("label"));

        horizontalLayout_2->addWidget(label);

        deviceComboBox = new QComboBox(EDAC40Panel);
        deviceComboBox->setObjectName(QString::fromUtf8("deviceComboBox"));
        deviceComboBox->setMinimumSize(QSize(151, 0));

        horizontalLayout_2->addWidget(deviceComboBox);

        refreshButton = new QPushButton(EDAC40Panel);
        refreshButton->setObjectName(QString::fromUtf8("refreshButton"));

        horizontalLayout_2->addWidget(refreshButton);


        verticalLayout_4->addLayout(horizontalLayout_2);

        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setSpacing(6);
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        label_2 = new QLabel(EDAC40Panel);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        horizontalLayout_4->addWidget(label_2);

        ipLabel = new QLabel(EDAC40Panel);
        ipLabel->setObjectName(QString::fromUtf8("ipLabel"));
        ipLabel->setEnabled(true);

        horizontalLayout_4->addWidget(ipLabel);


        verticalLayout_4->addLayout(horizontalLayout_4);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setSpacing(6);
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        groupBox = new QGroupBox(EDAC40Panel);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        verticalLayout = new QVBoxLayout(groupBox);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        valueUnitButton = new QRadioButton(groupBox);
        valueUnitButton->setObjectName(QString::fromUtf8("valueUnitButton"));
        valueUnitButton->setChecked(true);

        verticalLayout->addWidget(valueUnitButton);

        valueThreeQuartersButton = new QRadioButton(groupBox);
        valueThreeQuartersButton->setObjectName(QString::fromUtf8("valueThreeQuartersButton"));

        verticalLayout->addWidget(valueThreeQuartersButton);

        valueOneHalfButton = new QRadioButton(groupBox);
        valueOneHalfButton->setObjectName(QString::fromUtf8("valueOneHalfButton"));

        verticalLayout->addWidget(valueOneHalfButton);

        valueOneQuarterButton = new QRadioButton(groupBox);
        valueOneQuarterButton->setObjectName(QString::fromUtf8("valueOneQuarterButton"));

        verticalLayout->addWidget(valueOneQuarterButton);

        valueZeroButton = new QRadioButton(groupBox);
        valueZeroButton->setObjectName(QString::fromUtf8("valueZeroButton"));

        verticalLayout->addWidget(valueZeroButton);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setSpacing(6);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        valueOtherButton = new QRadioButton(groupBox);
        valueOtherButton->setObjectName(QString::fromUtf8("valueOtherButton"));

        horizontalLayout->addWidget(valueOtherButton);

        valueSpinBox = new QSpinBox(groupBox);
        valueSpinBox->setObjectName(QString::fromUtf8("valueSpinBox"));
        valueSpinBox->setEnabled(false);
        valueSpinBox->setMaximum(65535);
        valueSpinBox->setSingleStep(100);
        valueSpinBox->setValue(65535);

        horizontalLayout->addWidget(valueSpinBox);


        verticalLayout->addLayout(horizontalLayout);


        horizontalLayout_3->addWidget(groupBox);

        groupBox_2 = new QGroupBox(EDAC40Panel);
        groupBox_2->setObjectName(QString::fromUtf8("groupBox_2"));
        verticalLayout_2 = new QVBoxLayout(groupBox_2);
        verticalLayout_2->setSpacing(6);
        verticalLayout_2->setContentsMargins(11, 11, 11, 11);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        constantButton = new QRadioButton(groupBox_2);
        constantButton->setObjectName(QString::fromUtf8("constantButton"));
        constantButton->setChecked(true);

        verticalLayout_2->addWidget(constantButton);

        squareButton = new QRadioButton(groupBox_2);
        squareButton->setObjectName(QString::fromUtf8("squareButton"));

        verticalLayout_2->addWidget(squareButton);

        rampButton = new QRadioButton(groupBox_2);
        rampButton->setObjectName(QString::fromUtf8("rampButton"));

        verticalLayout_2->addWidget(rampButton);

        verticalSpacer = new QSpacerItem(20, 65, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_2->addItem(verticalSpacer);


        horizontalLayout_3->addWidget(groupBox_2);

        verticalLayout_3 = new QVBoxLayout();
        verticalLayout_3->setSpacing(6);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        commitButton = new QPushButton(EDAC40Panel);
        commitButton->setObjectName(QString::fromUtf8("commitButton"));
        commitButton->setEnabled(false);

        verticalLayout_3->addWidget(commitButton);

        stopButton = new QPushButton(EDAC40Panel);
        stopButton->setObjectName(QString::fromUtf8("stopButton"));
        stopButton->setEnabled(false);

        verticalLayout_3->addWidget(stopButton);


        horizontalLayout_3->addLayout(verticalLayout_3);


        verticalLayout_4->addLayout(horizontalLayout_3);

        groupBox_3 = new QGroupBox(EDAC40Panel);
        groupBox_3->setObjectName(QString::fromUtf8("groupBox_3"));
        gridLayout = new QGridLayout(groupBox_3);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        estimatedRangeLabel = new QLabel(groupBox_3);
        estimatedRangeLabel->setObjectName(QString::fromUtf8("estimatedRangeLabel"));
        estimatedRangeLabel->setEnabled(false);

        gridLayout->addWidget(estimatedRangeLabel, 1, 0, 1, 2);

        horizontalLayout_5 = new QHBoxLayout();
        horizontalLayout_5->setSpacing(6);
        horizontalLayout_5->setObjectName(QString::fromUtf8("horizontalLayout_5"));
        rangeMinLabel = new QLabel(groupBox_3);
        rangeMinLabel->setObjectName(QString::fromUtf8("rangeMinLabel"));
        rangeMinLabel->setEnabled(false);

        horizontalLayout_5->addWidget(rangeMinLabel);

        dotsLabel = new QLabel(groupBox_3);
        dotsLabel->setObjectName(QString::fromUtf8("dotsLabel"));

        horizontalLayout_5->addWidget(dotsLabel);

        rangeMaxLabel = new QLabel(groupBox_3);
        rangeMaxLabel->setObjectName(QString::fromUtf8("rangeMaxLabel"));
        rangeMaxLabel->setEnabled(false);

        horizontalLayout_5->addWidget(rangeMaxLabel);


        gridLayout->addLayout(horizontalLayout_5, 1, 2, 1, 2);

        horizontalSpacer_2 = new QSpacerItem(78, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout->addItem(horizontalSpacer_2, 1, 4, 1, 1);

        offsetDACLabel = new QLabel(groupBox_3);
        offsetDACLabel->setObjectName(QString::fromUtf8("offsetDACLabel"));
        offsetDACLabel->setEnabled(false);

        gridLayout->addWidget(offsetDACLabel, 2, 0, 1, 1);

        offsetDACSpinBox = new QSpinBox(groupBox_3);
        offsetDACSpinBox->setObjectName(QString::fromUtf8("offsetDACSpinBox"));
        offsetDACSpinBox->setEnabled(false);
        offsetDACSpinBox->setMinimumSize(QSize(61, 0));
        offsetDACSpinBox->setMaximum(16383);
        offsetDACSpinBox->setValue(8192);

        gridLayout->addWidget(offsetDACSpinBox, 2, 1, 1, 2);

        offsetDACSlider = new QSlider(groupBox_3);
        offsetDACSlider->setObjectName(QString::fromUtf8("offsetDACSlider"));
        offsetDACSlider->setEnabled(false);
        offsetDACSlider->setMaximum(16383);
        offsetDACSlider->setPageStep(100);
        offsetDACSlider->setValue(8192);
        offsetDACSlider->setOrientation(Qt::Horizontal);
        offsetDACSlider->setInvertedAppearance(true);
        offsetDACSlider->setInvertedControls(true);

        gridLayout->addWidget(offsetDACSlider, 2, 3, 1, 2);

        gainLabel = new QLabel(groupBox_3);
        gainLabel->setObjectName(QString::fromUtf8("gainLabel"));
        gainLabel->setEnabled(false);

        gridLayout->addWidget(gainLabel, 3, 0, 1, 1);

        gainSpinBox = new QSpinBox(groupBox_3);
        gainSpinBox->setObjectName(QString::fromUtf8("gainSpinBox"));
        gainSpinBox->setEnabled(false);
        gainSpinBox->setMinimumSize(QSize(61, 0));
        gainSpinBox->setMaximum(65535);
        gainSpinBox->setValue(65535);

        gridLayout->addWidget(gainSpinBox, 3, 1, 1, 2);

        gainSlider = new QSlider(groupBox_3);
        gainSlider->setObjectName(QString::fromUtf8("gainSlider"));
        gainSlider->setEnabled(false);
        gainSlider->setMaximum(65535);
        gainSlider->setPageStep(100);
        gainSlider->setValue(65535);
        gainSlider->setOrientation(Qt::Horizontal);

        gridLayout->addWidget(gainSlider, 3, 3, 1, 2);

        adjustmentsEnableCheckBox = new QCheckBox(groupBox_3);
        adjustmentsEnableCheckBox->setObjectName(QString::fromUtf8("adjustmentsEnableCheckBox"));
        adjustmentsEnableCheckBox->setEnabled(false);

        gridLayout->addWidget(adjustmentsEnableCheckBox, 4, 0, 1, 1);

        resetDefaultsButton = new QPushButton(groupBox_3);
        resetDefaultsButton->setObjectName(QString::fromUtf8("resetDefaultsButton"));
        resetDefaultsButton->setEnabled(false);

        gridLayout->addWidget(resetDefaultsButton, 4, 1, 1, 2);

        saveDefaultsButton = new QPushButton(groupBox_3);
        saveDefaultsButton->setObjectName(QString::fromUtf8("saveDefaultsButton"));
        saveDefaultsButton->setEnabled(false);

        gridLayout->addWidget(saveDefaultsButton, 4, 3, 1, 2);


        verticalLayout_4->addWidget(groupBox_3);


        retranslateUi(EDAC40Panel);
        QObject::connect(refreshButton, SIGNAL(clicked()), EDAC40Panel, SLOT(listDevices()));
        QObject::connect(valueOtherButton, SIGNAL(toggled(bool)), valueSpinBox, SLOT(setEnabled(bool)));
        QObject::connect(deviceComboBox, SIGNAL(activated(int)), EDAC40Panel, SLOT(chooseDevice(int)));
        QObject::connect(commitButton, SIGNAL(clicked()), EDAC40Panel, SLOT(commitChanges()));
        QObject::connect(stopButton, SIGNAL(clicked()), EDAC40Panel, SLOT(stopGenerator()));
        QObject::connect(adjustmentsEnableCheckBox, SIGNAL(toggled(bool)), EDAC40Panel, SLOT(setAdjustmentsEnabled(bool)));
        QObject::connect(offsetDACSpinBox, SIGNAL(valueChanged(int)), offsetDACSlider, SLOT(setValue(int)));
        QObject::connect(offsetDACSlider, SIGNAL(valueChanged(int)), offsetDACSpinBox, SLOT(setValue(int)));
        QObject::connect(gainSpinBox, SIGNAL(valueChanged(int)), gainSlider, SLOT(setValue(int)));
        QObject::connect(gainSlider, SIGNAL(valueChanged(int)), gainSpinBox, SLOT(setValue(int)));
        QObject::connect(offsetDACSpinBox, SIGNAL(valueChanged(int)), EDAC40Panel, SLOT(sendAdjustments()));
        QObject::connect(gainSpinBox, SIGNAL(valueChanged(int)), EDAC40Panel, SLOT(sendAdjustments()));
        QObject::connect(offsetDACSpinBox, SIGNAL(valueChanged(int)), EDAC40Panel, SLOT(calculateRange()));
        QObject::connect(gainSpinBox, SIGNAL(valueChanged(int)), EDAC40Panel, SLOT(calculateRange()));
        QObject::connect(resetDefaultsButton, SIGNAL(clicked()), EDAC40Panel, SLOT(setDefaultAdjustments()));
        QObject::connect(saveDefaultsButton, SIGNAL(clicked()), EDAC40Panel, SLOT(saveDefaults()));
        QObject::connect(valueUnitButton, SIGNAL(toggled(bool)), EDAC40Panel, SLOT(updateVoltageValue()));
        QObject::connect(valueThreeQuartersButton, SIGNAL(toggled(bool)), EDAC40Panel, SLOT(updateVoltageValue()));
        QObject::connect(valueOneHalfButton, SIGNAL(toggled(bool)), EDAC40Panel, SLOT(updateVoltageValue()));
        QObject::connect(valueOneQuarterButton, SIGNAL(toggled(bool)), EDAC40Panel, SLOT(updateVoltageValue()));
        QObject::connect(valueZeroButton, SIGNAL(toggled(bool)), EDAC40Panel, SLOT(updateVoltageValue()));
        QObject::connect(valueOtherButton, SIGNAL(toggled(bool)), EDAC40Panel, SLOT(updateVoltageValue()));
        QObject::connect(valueSpinBox, SIGNAL(valueChanged(int)), EDAC40Panel, SLOT(updateVoltageValue()));

        QMetaObject::connectSlotsByName(EDAC40Panel);
    } // setupUi

    void retranslateUi(QDialog *EDAC40Panel)
    {
        EDAC40Panel->setWindowTitle(QApplication::translate("EDAC40Panel", "EDAC40", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("EDAC40Panel", "Choose device:", 0, QApplication::UnicodeUTF8));
        refreshButton->setText(QApplication::translate("EDAC40Panel", "Refresh list", 0, QApplication::UnicodeUTF8));
        label_2->setText(QApplication::translate("EDAC40Panel", "Device IP address:", 0, QApplication::UnicodeUTF8));
        ipLabel->setText(QApplication::translate("EDAC40Panel", "XXX.XXX.XXX.XXX", 0, QApplication::UnicodeUTF8));
        groupBox->setTitle(QApplication::translate("EDAC40Panel", "Amplitude", 0, QApplication::UnicodeUTF8));
        valueUnitButton->setText(QApplication::translate("EDAC40Panel", "1", 0, QApplication::UnicodeUTF8));
        valueThreeQuartersButton->setText(QApplication::translate("EDAC40Panel", "3/4", 0, QApplication::UnicodeUTF8));
        valueOneHalfButton->setText(QApplication::translate("EDAC40Panel", "1/2", 0, QApplication::UnicodeUTF8));
        valueOneQuarterButton->setText(QApplication::translate("EDAC40Panel", "1/4", 0, QApplication::UnicodeUTF8));
        valueZeroButton->setText(QApplication::translate("EDAC40Panel", "0", 0, QApplication::UnicodeUTF8));
        valueOtherButton->setText(QApplication::translate("EDAC40Panel", "Other:", 0, QApplication::UnicodeUTF8));
        groupBox_2->setTitle(QApplication::translate("EDAC40Panel", "Shape", 0, QApplication::UnicodeUTF8));
        constantButton->setText(QApplication::translate("EDAC40Panel", "Constant", 0, QApplication::UnicodeUTF8));
        squareButton->setText(QApplication::translate("EDAC40Panel", "Square", 0, QApplication::UnicodeUTF8));
        rampButton->setText(QApplication::translate("EDAC40Panel", "Ramp", 0, QApplication::UnicodeUTF8));
        commitButton->setText(QApplication::translate("EDAC40Panel", "Go", 0, QApplication::UnicodeUTF8));
        stopButton->setText(QApplication::translate("EDAC40Panel", "Stop", 0, QApplication::UnicodeUTF8));
        groupBox_3->setTitle(QApplication::translate("EDAC40Panel", "Adjustments", 0, QApplication::UnicodeUTF8));
        estimatedRangeLabel->setText(QApplication::translate("EDAC40Panel", "Estimated range (V):", 0, QApplication::UnicodeUTF8));
        rangeMinLabel->setText(QApplication::translate("EDAC40Panel", "-12.00", 0, QApplication::UnicodeUTF8));
        dotsLabel->setText(QApplication::translate("EDAC40Panel", "...", 0, QApplication::UnicodeUTF8));
        rangeMaxLabel->setText(QApplication::translate("EDAC40Panel", "+12.00", 0, QApplication::UnicodeUTF8));
        offsetDACLabel->setText(QApplication::translate("EDAC40Panel", "Offset:", 0, QApplication::UnicodeUTF8));
        gainLabel->setText(QApplication::translate("EDAC40Panel", "Gain:", 0, QApplication::UnicodeUTF8));
        adjustmentsEnableCheckBox->setText(QApplication::translate("EDAC40Panel", "Enable", 0, QApplication::UnicodeUTF8));
        resetDefaultsButton->setText(QApplication::translate("EDAC40Panel", "Restore Defaults", 0, QApplication::UnicodeUTF8));
        saveDefaultsButton->setText(QApplication::translate("EDAC40Panel", "Save to NVRAM", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class EDAC40Panel: public Ui_EDAC40Panel {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_EDAC40PANEL_H
