import threading
import sys
import time
from networktables import NetworkTables

if len(sys.argv) != 2:
    print("You must supply the ip address of the RoboIO in the 10.xx.xx.2 form")
    exit(0)

cond = threading.Condition()
notified = [False]

def connectionListener(connected, info):
    print(info, '; Connected=%s' % connected)
    with cond:
        notified[0]=True
        cond.notify()

ip = sys.argv[1]
NetworkTables.initialize(server=ip)
NetworkTables.addConnectionListener(connectionListener, immediateNotify=True)

with cond:
    print("Waiting")
    if not notified[0]:
        cond.wait()

print("Connected!")

table = NetworkTables.getTable("SmartDashboard")

i = 0
while True:
#    print("RobotTime:", table.getNumber("robotTime", -1))

#    table.putNumber("JetsonTime", i)
    print("JetsonTime:", table.getNumber("JetsonTime", -1))

    time.sleep(1)

    i+=1


