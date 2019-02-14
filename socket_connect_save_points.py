import socket
import time
import numpy as np
import pickle


# define the host and port
HOST = "192.168.0.100"   # The remote host
PORT = 30002           # The same port as used by the server

# establish socket connection
s = socket.socket() 
s.bind((HOST, PORT))
s.listen(5)
c, addr = s.accept()

# the list of all cell positions coordinates on pallet
Pattern_Pose = []

# the list of all approah and retreat offset for each cell on pallet
app_ret_offset = []


count = 1

while count < 33:

    print(count)
    msg = c.recv(1024)
    print(msg)
    time.sleep(0.5)

    msg = c.recv(1024)
    print ('The approach position is(XYZ vector)')
    print (msg)
    print(type(msg))
    approach_data = msg.decode('utf8').replace("'" , '"')
    print(approach_data)
    #print(type(approach_data))
    #print('----', approach_data.split(","))
    #print('----', approach_data.split(",")[2])


    time.sleep(0.5)
    msg = c.recv(1024)
    print ('The approach position is(Joint angle)')
    print (msg)


    msg = c.recv(1024)
    print(msg)
    time.sleep(0.5)

    msg = c.recv(1024)
    print ('The robot position is(XYZ vector)')
    print (msg)
    pose_data = msg.decode('utf8').replace("'" , '"')
    # add coordinates to list
    Pattern_Pose.append(pose_data)
    # calculate the offset
    offset = float(approach_data.split(",")[2]) - float(pose_data.split(",")[2])
    # add the offset to list
    app_ret_offset.append(offset)

    print('offset is ' + str(offset))


    time.sleep(0.5)
    msg = c.recv(1024)
    print ('The robot position is(Joint angle)')
    print (msg)
    

    count += 1

print('-------------')
print(Pattern_Pose)
print(type(Pattern_Pose)) 
print('-------------')
print(Pattern_Pose[0])
print('-------------')
print(Pattern_Pose[0].split(","))
print(Pattern_Pose[0].split(",")[2])
print(type(float(Pattern_Pose[0].split(",")[2])))
print('-------------')
print(app_ret_offset)
print(np.mean(app_ret_offset)) 
print(round(np.mean(app_ret_offset),6))


s.close()
