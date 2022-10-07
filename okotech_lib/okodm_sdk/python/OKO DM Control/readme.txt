Installation and configuration of OKO DM Control
------------------------------------------------

1) Start "setup.exe" to install OKO DM Control.

2) To be able to operate with deformable mirrors, you also need to
install "DLPortIO" library. Start "port95nt.exe" to install it.
Restart your computer when the installation is complete.

3) Configuration of a deformable mirror. OKO's deformable mirror can
be interfaced using OKO's ISA and PCI digital boards and USB units.
Configuration files are supplied for the following configurations.

"mmdm37isa.txt" - 37-channel micromachined membrane deformable
mirror, 2 ISA boards

"mmdm37pci.txt" - 37-channel micromachined membrane deformable
mirror, 2 PCI boards

"mmdm37usb.txt" - 37-channel micromachined membrane deformable
mirror, 1 USB unit

"mmdm59pci.txt" - 59-channel micromachined membrane deformable
mirror, 3 PCI boards

"mmdm79pci.txt" - 79-channel micromachined membrane deformable
mirror, 4 PCI boards

"mmdm79_40usb.txt" - 79-channel micromachined membrane deformable
mirror with 40 mm aperture, 2 USB units

"piezo19pci.txt" - 19-channel deformable mirror with piezoelectric
control, 1 PCI board

"piezo19usb.txt" - 19-channel deformable mirror with piezoelectric
control, 1 USB unit

"piezo_lin20usb.txt" - 20-channel linear deformable mirror with 
piezoelectric control, 1 USB unit

"piezo37pci.txt" - 37-channel deformable mirror with piezoelectric
control, 2 PCI boards

"piezo37usb.txt" - 37-channel deformable mirror with piezoelectric
control, 1 USB unit

"piezo37_2005usb.txt" - 37-channel deformable mirror with
piezoelectric control (design of 2005), 1 USB unit

"piezo109usb.txt" - 109-channel deformable mirror with piezoelectric
control, 3 USB units


To load configuration of channels for your deformable mirror, start
"OKO DM Control" and press "Configuration". In the dialog box
"Mirror interface", press "Configure", then "Load" button and load a
proper configuration file from the installation directory.

IMPORTANT! After loading the configuration file, you need to correct
base addresses of the boards (for PCI and ISA configurations) or
serial numbers of the controllers (for USB configurations). See the
instructions below.

For ISA boards
--------------

Base address of each board can be set by the address switch
positioned at the board. Jumper combinations corresponding to
certain addresses are given in the manual for deformable mirrors.

For PCI boards
--------------

PCI boards are plug-and-play compliant; their adrresses are assigned
dynamically. You need to install drivers of these boards from the CD
supplied with the system. To get the addresses, go to "Control
Panel->System->Hardware->Device manager". The boards are listed in
the section "Multifunction adapters". Double-click on each device
named "PROTO-3/PCI" and check the section "Resources" for its base
I/O address.

For USB units
-------------

Serial number of a USB unit is written on the label attached to the
unit's cover. You also need to install drivers from the CD supplied
with the system.

Correct the addresses/serial numbers in the dialog box "Deformable
mirror configuration" and press "OK".

4) Your mirror is now configured. You may press "Control" button and
manually change voltages applied to the actuators. The voltage
values are normalized to the range -1..1, where -1 corresponds to
zero voltage, and 1 to the maximum one.

INPORTANT! For MMDM configurations, the voltage applied to the
mirror changes as a square root of the control parameter, because
the mirror response depends quadratically on the applied voltage.
For PDMs this dependence is linear. You may choose between linear
and quadratic response type in the dialog box "Deformable mirror
configurations".
