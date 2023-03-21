from datetime import datetime, timedelta

import time

import requests

import intel_jtag_uart

import matplotlib.pyplot as plt

import numpy as np

DATA_BATCH_SIZE = 100 # if number of data points collected reaches this point then send them as one batch to AWS server.

BUFFER_SIZE_PANIC = 500 # if buffer gets beyond 500 data points. Can't keep up! Is the server overloaded?

DATA_BATCH_INTERVAL = timedelta(seconds=2) # Send at most every 2 seconds

WARN_NO_DATA_AFTER = timedelta(seconds=5) # If no data received in the last five seconds, warn

def upconvert_bytes(the_bytes, signed: bool):
    #print(f"{the_bytes[0]:b}")

    out = 0

    iteration = 0
    for b in the_bytes[::-1]:
        out |= b << (7*iteration)
        iteration += 1
    
    # out = out << 4
    if signed and the_bytes[0] & 0b01000000: # if signed and payload MSB high, sign extend
        #return out - 2**(8*len(the_bytes))
        return out - 2 ** 14
    else:
        return out

if __name__ == "__main__":
    jtag = intel_jtag_uart.intel_jtag_uart()
    jtag.flush()
    data_batch = []
    xpos = []
    ypos = []
    t = []
    tcur = 0

    buffer = b""

    last_upload_time = datetime.min

    last_fpga_contact = datetime.now()
    start = time.time()

    try:
        while True:
            new_chars = jtag.read()
            if len(new_chars) == 0:
                if datetime.now() - last_fpga_contact > WARN_NO_DATA_AFTER:
                    print(f"Warning: no data received for {datetime.now() - last_fpga_contact}")
                    last_fpga_contact = datetime.now()
            else:
                last_fpga_contact = datetime.now()

            buffer += new_chars

            if b"\xff" in buffer:
                data = buffer.split(b"\xff", 1)[0]
                buffer = buffer.split(b"\xff", 1)[1]
                if len(buffer) > BUFFER_SIZE_PANIC:
                    buffer = b""
                    print("Buffer exceeded, resetting")
                    continue
                if len(data) < 1:
                    continue
                if (data[0] & 0b11000000) == 0b11000000: # Special header received
                    sgn = data[0] & 0b00100000 # represents signed or not
                    point_length = data[0] & 0b00011111 # last 5 bits represent size
                    data = data[1:]
                    
                    if len(data) % point_length != 0:
                        print(f"Error: buffer was not aligned with the header-provided data point size!"\
                            f"(header `{data[0]:b}` gave data point length of {point_length}, but data had length {len(data)}")
                        continue
                    for i in range(len(data)//point_length):
                        if i == 0:
                            xpos.append(-upconvert_bytes(data[i:i+point_length], 1) / 10)
                            # print("steps " + str(upconvert_bytes(data[i:i+point_length], 0)))
                            t.append(tcur)
                        elif i == 1:
                            ypos.append(-upconvert_bytes(data[i*point_length:(i+point_length)*point_length], 1) / 10)
                            # print("magnet " + str(upconvert_bytes(data[i*point_length:(i+point_length)*point_length], 0)))
                        tdif = (datetime.now() - last_fpga_contact).microseconds / 1e3
                        while tdif == 0:
                            tdif = (datetime.now() - last_fpga_contact).microseconds / 1e3
                        tcur += tdif

                    # print(data)
                    # if len(data_batch) >= DATA_BATCH_SIZE or datetime.now() - last_upload_time > DATA_BATCH_INTERVAL:
                    #     print(f"placeholder: uploading some data {data_batch[0]}")
                    #     data_batch = []
                    #     last_upload_time = datetime.now()
    except KeyboardInterrupt:
        fig, ax = plt.subplots()
        # np.save('steps', np.array(xpos))
        # np.save('magnt', np.array(ypos))
        # np.save('times', np.array(t))

        xposNP = np.array(xpos[10:])
        yposNP = np.array(ypos[10:])
        tNP = np.array(t[10:])

        maxsize = max(xposNP.shape[0], yposNP.shape[0], tNP.shape[0])

        if(xposNP.shape[0] != maxsize):
            xposNP = np.append(xposNP, 0)
        if(yposNP.shape[0] != maxsize):
            yposNP = np.append(yposNP, 0)
        if(tNP.shape[0] != maxsize):
            tNP = np.append(tNP, 0)

        # ax.plot(tNP, yposNP)
        # ax.plot(tNP, xposNP)
        ax.set_aspect('equal', adjustable='box')
        ax.plot(xposNP[0], yposNP[1], "rx")
        ax.plot(xposNP, yposNP)
        plt.show()

        combined = np.c_[tNP, xposNP, yposNP]

        np.save('cozzyData_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), combined)

        # combined.tofile('data')

        # input("Press Enter to exit")

