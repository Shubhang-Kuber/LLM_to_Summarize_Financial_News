#Step-1
#Requests is a class that has built in methods to make HTTP requests to the desired URL.
#Requests like GET, PUT, POST, PATCH can be used
%pip install requests


#Step-2
#Beatiful Soup is a web scrapping framework of Python
#It is the process of extracting data from the website to automate information transfer
#It is used for parseing and navigating html pages
%pip install bs4

#Step-3
#Import all the required libraries
from bs4 import BeautifulSoup as BS
import requests as req


#Step-4
#We will use the websiete of businesstoday.in for our scrapping
#Right click on the news and inspect 
#Check which anchor tag is being used for the news tabs
#We wil use that tag name in our code for scrapping purposes


#Step-5
# Importing all the required libraries
from bs4 import BeautifulSoup as BS
import requests as req

url = "https://www.businesstoday.in/latest/economy"

#From the requests class; we are sending a "Get Request" to the webpge
#Variable webpage will store the entire info of the webpage
webpage = req.get(url)  

# HTML Parser refers to the entire html page
trav = BS(webpage.content, "html.parser")


# find_all is used to find all the anchor tags in the page
#link.string is used to extract all the text info from the webpage
for link in trav.find_all('a'):
    print(type(link.string), " ", link.string)


# Step 5
#We will be able to see two types of outputs
#First is the "None"
#Second is the "bs4.element.NavigableString"
#We wil set the character lenght to maximum of 50 characters


#Step-6
# IMPORT ALL THE REQUIRED LIBRARIES
from bs4 import BeautifulSoup as BS
import requests as req

url = "https://www.businesstoday.in/latest/economy"

webpage = req.get(url)
trav = BS(webpage.content, "html.parser")
#i is a counter variable similar to that of the for loop
i = 1
for link in trav.find_all('a'):
  
    #link.string will give the length inside the anchor
    #Checks the datatype and compares with the bs4.element.NavigableString\
    #len() will check the length of the string
    if(str(type(link.string)) == "<class 'bs4.element.NavigableString'>"
       and len(link.string) > 25):

        print(str(i)+".", link.string)
        #Incrementing the variable i of the loop
        i += 1
