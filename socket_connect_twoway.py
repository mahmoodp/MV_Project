'''
import socket

#Class to define a robot in a 3D environment
class Socket:

    #Constructor of the class
    def __init__(self):
        #socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #socket.bind(('', 15555))

        #while True:
        #    socket.listen(5)
        #    client, address = socket.accept()
        #    print ("{} connected".format( address ))
        #    response = client.recv(255)
        #    if response != "":
        #        print (response)
        #print ("Close")
        #client.close()
        #stock.close()
        buf=1024
        hote = "192.168.1.25"
        port = 30000
        connected = True
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((hote, port))
        print ("Connection on {}".format(port))

        sock.send(b"Hello!")
        while connected:
            try:
                # réception et affichage de la réponse
                reponse, adr = sock.recvfrom(buf)
                print ("=> %s" % reponse)
            except:
                print ("connection closed")
                connected = False

        print ("Close")
        sock.close()

'''

import socket
import time
import numpy as np



HOST = "192.168.0.100"   # The remote host

#HOST = socket.gethostbyname(socket.gethostname())
#if HOST.startswith('127.0.'):
#    HOST = '0.0.0.0'


PORT = 30002           # The same port as used by the server
#connected = True


s = socket.socket() 
s.bind((HOST, PORT))
s.listen(5)
c, addr = s.accept()

# the list of all cell positions coordinates on pallet
Pattern_Pose = []

# the list of all approah and retreat offset for each cell on pallet
app_ret_offset = []

count = 1

while count < 3:

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

#s.send(b"set_digital_out(2,False)" + "\n".encode('ascii'))
#s.send (b"movej([-0.5405182705025187, -2.350330184112267, -1.316631037266588, -2.2775736604458237, 3.3528323423665642, -1.2291967454894914], a=1.3962634015954636, v=1.0471975511965976)" + "\n".encode('ascii'))

    
    # Move to Barcode Scan position
    #s.send (b"movej([0.7871907692279203, -1.5103761085131069, -2.2033988364794155, -1.024165455495016, 1.5907973051071167, 0.7398521304130554], a=1.3962634015954636, v=1.0471975511965976)" + "\n".encode('ascii'))

    #time.sleep(4)

    # Move to Barcode Camera position
    #s.send (b" movej([0.41085015533560754, -1.2924395451082678, -1.8405188490976556, -1.5889028464373958, 1.58100740589828, 0.4406203150331578], a=1.3962634015954636, v=1.0471975511965976)" + "\n".encode('ascii'))


    

# Move to test position
#s.send (b" movej([0.123849,-1.39692,-2.29239,-0.986967,1.54953,0.0980444], a=1.3962634015954636, v=1.0471975511965976)" + "\n".encode('ascii'))




    #print ("Connection on {}".format(PORT))
    #data = s.recv(1024)
    #data = data.decode('utf-8')
  
    #print(data)
    
    #if data == "current_pos":
    #    print ("Received", repr(data))
    #print ("Received", repr(data))

    #if data == "hello":
    #    print ("hi")




'''
msg = c.recv(1024)
print ('The robot position is(XYZ vector)')
print (msg)

time.sleep(0.5)
msg = c.recv(1024)
print ('The robot position is(Joint angle)')
print (msg)
'''


s.close()
