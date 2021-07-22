import os

def menu():

   r = input("Where you want to Execute your program(local/remote) : ")
   print(r)


   if r == "remote":
     ip = input("Enter IP Address :")


   while True:
	
        
        if r == "local":
                print("""
                      \n
	              Press 1: to run date cmd 
	              Press 2: to Calander
	              Press 3: Launch Youtube
	              Press 4: Launch Qr/BarCode Scanner
                      Press 5: Open Chrome
                      Press 6: Launch MxPlayer
                      Press 7: To exist
                      """) 
                ch = input("Enter ur choice: ")
                print(ch)

                if int(ch) == 1:
			#ln.linux_cmd()
	                 os.system("date /t")
                elif int(ch) == 2:
                         os.system("cal")
                elif int(ch) == 3:                       
                         webbrowser.open('start www.youtube.com')
                elif int(ch) == 4:
                         os.system("python qr.py")
                elif int(ch) == 5:
                         webbrowser.open('chrome')	
                elif int(ch) == 6:
                         os.system("python3 docker.py")		
                elif int(ch) == 7:
                         os.system("ansible-playbook web.yml")		
                elif int(ch) == 8:
                         os.system("ansible-playbook ec2.yml")	
                elif int(ch) == 9:
                         os.system("ansible-playbook haproxy.yml")
                else:
                         exit()
       
	
        elif r == "remote":
		#ip = input("Enter remote ip:")  

                
                print("""
         	\n
	        Press 1: to run Linux basic cmd
         	Press 2: to cal
	        Press 3: reboot
	        Press 4: show list
                Press 5: Configure  Docker
                Press 6: Docker Functions
                Press 7: Launch Webserver 
                Press 8: to launch AWS EC2 Instance
                Press 9: SetUp LoadBalancer
	        Press 10: to exit
                 """)
                ch = input("Enter ur choice: ")
                print(ch)
   
		       
                if int(ch) == 1:
                        os.system("ssh root@{}  date".format(ip))
                elif int(ch) == 2:
                        os.system("ssh root@{} cal".format(ip))
                elif int(ch) == 3:
                        os.system("ssh root@ {} reboot".format(ip))
            
                elif int(ch) == 4:
                        os.system("ssh root@{} ls".format(ip))

                elif int(ch) == 5:
                        os.system("ssh root@{} ls".format(ip))
                elif int(ch) == 6:
                        os.system("ssh root@{}  python3 /root/arth-ws/docker.py".format(ip))

                elif int(ch) == 7:
                       os.system("ssh {} ".format(ip))
		
                elif int(ch) == 8:
                      os.system("ssh root@{} ansible-playbook /root/arth-ws/ec2.yml ".format(ip))
		
                elif int(ch) == 9:
                     os.system("ssh root@{} ansible-playbook /root/arth-ws/haproxy.yml ".format(ip))
                else:
                      exit()
        else: 	
             print("condition don't support")		









import getpass

os.system("tput setaf 3")
print("\t\t\t Welcome to my Menu !!")
os.system("tput setaf 7")
print("\t\t\t-----------------------")


passwd = getpass.getpass("Enter your passwd:")

if passwd != "lw":
   print("password is incorrect.....")
   exit()




import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
print(cv2.__version__)
# Get the training data we previously made
data_path = 'd://faces//'
# a=listdir('d:/faces')
# print(a)
# """
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

# Create arrays for training data and labels
Training_Data, Labels = [], []

# Open training images in our datapath
# Create a numpy array for training data
for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)
# 
# Create a numpy array for both training data and labels
Labels = np.asarray(Labels, dtype=np.int32)
model=cv2.face_LBPHFaceRecognizer.create()
# Initialize facial recognizer
# model = cv2.face_LBPHFaceRecognizer.create()
# model=cv2.f
# NOTE: For OpenCV 3.0 use cv2.face.createLBPHFaceRecognizer()

# Let's train our model 
model.train(np.asarray(Training_Data), np.asarray(Labels))
print("Model trained sucessefully")




   


  



import cv2
import numpy as np
import webbrowser


face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_detector(img, size=0.5):
    
    # Convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img, []
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi


# Open Webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    
    image, face = face_detector(frame)
    
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Pass face to prediction model
        # "results" comprises of a tuple containing the label and the confidence value
        results = model.predict(face)
        print(results)
        if results[1] < 500:
            confidence = int( 100 * (1 - (results[1])/400) )
            display_string = str(confidence) + '% Confident it is User'
            
        cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)
        
        if confidence > 80:
            cv2.putText(image, "Hey Bro ", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Face Recognition', image )
            #webbrowser.open('')
            break
            
            
        else:
            cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.imshow('Face Recognition', image )

    except:
        cv2.putText(image, "No Face Found", (220, 120) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
        cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
        cv2.imshow('Face Recognition', image )
        pass
        
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows()

menu()






