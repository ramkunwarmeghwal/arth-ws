import os

os.system("yum install httpd -y")


os.system("cp hello.html   /var/www/html/ ")


os.system("systemctl start httpd")


print("weserver is ready")
