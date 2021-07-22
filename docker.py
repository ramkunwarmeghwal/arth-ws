import os

#os.system("setenforce 0")
#os.system("systemctl stop firewalld")
#os.system("systemctl restart docker.service")
#os.system("systemctl enable docker.service")



os.system("tput setaf 3")
print("\t\t\t Welcome to my Menu !!")
os.system("tput setaf 7")
print("\t\t\t-----------------------")



while True:

	print(""""
     	Press 0: For seeing all docker images
     	Press 1: For launch new docker
     	Press 2: For stop doker
     	Press 3: For start docker
     	Press 4: For see running all docker
    	Press 5: For terminate all docker 
	Press 6: For exit
	""")
	ch  = input("Enter Your Choice:")
	print(ch)


	if ch == "0":
		os.system("docker images")

	elif ch == "1":
		name = input("Enter docker name :")
		print(name)
		image = input("Enter image name like Centos/Ubuntu :")
		print(image)
		os.system("docker run -dit --name {}  {}:latest".format(name ,image))
	
	elif ch == "2":
       		name = input("Enter docker name :")
			#print(name)
       		os.system("docker stop {}".format(name))
       		print("docker stoped")
	elif ch == "3":
       		name = input("Enter docker name :")
		#print(name)
       		os.system("docker stop {}".format(name))
       		print("docker started")
	elif ch == "4":
       		os.system("docker ps -a ")
		
	elif ch == "5":
       		os.system("docker rm -f $(docker ps -a -q)")


	elif ch == "6":
                exit()

	else: 	
       		print("Your choice is not supported plz try again")     
   
#print(docker ps -a)
	
	
	
	
