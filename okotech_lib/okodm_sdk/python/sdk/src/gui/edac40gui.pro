CONFIG += static

QT       += core gui

INCLUDEPATH += ..
VPATH += ..

TARGET = edac40gui
TEMPLATE = app


SOURCES += main.cpp edac40panel.cpp edacgenerator.cpp edac40.c
HEADERS  += edac40panel.h  edacgenerator.h edac40.h
FORMS    += edac40panel.ui

win32 {
  LIBS  += -lws2_32
}