#!/usr/bin/python3
import time
import serial
import sys

# Paul Amoruso
# Group 7
# Senior Design 2
# Instructures: Lei Wei, Samuel Richie, and Dr. Kar
# UART code that on the Nano.

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
    
    # If we recieve the letter H then the pcb needes to be notified to close the hatch.
    if len(sys.argv) > 1 and sys.argv[1] == 'H':
        serial_port.write("H".encode())
        print (sys.argv[1])

    # If we recieve the letter U then the pcb needes to be notified to check ultrasonic sensor.
    elif len(sys.argv) > 1 and sys.argv[1] == 'U':
        serial_port.write("U".encode())
        print (sys.argv[1])

    # If we recieve the letter S then the pcb needes to be notified to play noise from speaker.
    elif len(sys.argv) > 1 and sys.argv[1] == 'S':
        serial_port.write("S".encode())
        print (sys.argv[1])

    # Write to the port when a squirl is detected or whatever the reason why.
    else:
        serial_port.write("A".encode())

    #serial_port.write("NVIDIA Jetson Nano Developer Kit\r\n".encode())
    Measurment = '0'
    while True:
        if serial_port.inWaiting() > 0:
            # Read what is being sent by the board.
            data = serial_port.read()
            # We now are able to decode the data recieved from MSP.
            data = data.decode()
            # Check if we are going to get ultrasonic measurements.
            if sys.argv[1] == 'U':
                #data = Integer.parseInt(data)
                # See if it is a number and start concatenating the string into a number
                if data.isnumeric():
                    Measurment =  Measurment + data
                # If there is a letter e sent, then that means the measurment is done and we can process the value.
                elif data == 'e':
                    print(int(Measurment))
            # Print to the console. (unnecessary, used just for a visual)
            print(data)
            # Write to the port whatever whever was recieved by the board.
            #serial_port.write(data)

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
