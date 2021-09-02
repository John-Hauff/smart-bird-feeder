#!/usr/bin/python3
import time
import serial

print("UART Demonstration Program")
print("NVIDIA Jetson Nano Developer Kit")

# Set up the serial port connection.
serial_port = serial.Serial(
    port="/dev/ttyTHS1",
    baudrate=9600,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
)
# Wait a second to let the port initialize
time.sleep(1)

try:
    # Send a simple header
    #serial_port.write("UART Demonstration Program\r\n".encode())

    # Write to the port when a squirl is detected or whatever the reason why.
    serial_port.write("A".encode())

    #serial_port.write("NVIDIA Jetson Nano Developer Kit\r\n".encode())
    while True:
        if serial_port.inWaiting() > 0:
            # Read what is being sent by the board.
            data = serial_port.read()
            # Print to the console. (unnecessary, used just for a visual)
            print(data)
            # Write to the port whatever whever was recieved by the board.
            serial_port.write(data)

            # If we get a carriage return, add a line feed too
            # \r is a carriage return; \n is a line feed
            # This is to help the tty program on the other end
            # Windows is \r\n for carriage return, line feed
            # Macintosh and Linux use \n
            if data == "\r".encode():
                # For Windows boxen on the other end
                serial_port.write("\n".encode())


except KeyboardInterrupt:
    print("Exiting Program")

except Exception as exception_error:
    print("Error occurred. Exiting Program")
    print("Error: " + str(exception_error))

finally:
    serial_port.close()
    pass
