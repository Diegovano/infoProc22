# infoProc22 Information Processing 2022-2023 Year 2 Group Project

## Oliver Cosgrove, Corey O'Malley, Patrick Beart, Chang Liu, Anastasis Varvarigos & Diego Van Overberghe

To set up the FPGA for I2C:

- download the extract the contents of `i2c_opencores_ollie_version.zip` to `C:\intelFPGA_lite\20.1\ip`
- `i2c_opencores_ollie_version.zip` can be found in `QUARTUS/`

If you are pulling and are having issues, try the following troubleshooting steps:

- Open Platform Designer, make sure there are components such as SPI, accelerometer SPI, timer, led_timer, led, UART and other.

- Assign Base Addresses
- Generate HDL
- Save .qsys file
- Check top level Verilog file has correct NIOS II connections
- Compile the project
- Navigate to `software/infoProc22_sw`, start the NIOS II shell.
- Run `nios2-bsp-generate-files(you may need to append .exe) --settings=../infoProc22_sw_bsp/settings.bsp --bsp-dir=../infoProc22_sw_bsp`
- Run `nios2-configure-sof -C ../..`
- Run `make download-elf`
- Finally, run `niso2-terminal`

If you have further issues, ask others.
