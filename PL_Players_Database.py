import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from bs4 import BeautifulSoup
import pandas as pd
import requests
import re
import numpy as np


def initial_scrape():

    driver = webdriver.Edge(executable_path = 'C://Users/rory/Downloads/edgedriver_win64/msedgedriver.exe')
    driver.get("https://www.premierleague.com/players?se=274&cl=-1")

    ScrollNumber = 15
    for i in range(1,ScrollNumber):
        driver.execute_script("window.scrollTo(1,50000)")
        time.sleep(5)

    file = open('DS.html', 'w')
    file.write(driver.page_source)
    file.close()

    driver.close()

    name_list = []
    link_list = []
    file = open('DS.html', 'r')
    soup = BeautifulSoup(file, 'html.parser')
    PNames = soup.find_all("a" ,{"class" : "playerName"})
    for name in PNames:
        name_list.append(name.text)

    name_frame = pd.DataFrame(name_list, columns = ["Name"])

    for link in  PNames:
        link_list.append("https:" + str(link.get('href')))

    link_frame = pd.DataFrame(link_list, columns = ['url'])
    DataBase = pd.concat([name_frame, link_frame], axis = 1)
    print (DataBase)
    DataBase.to_csv("DataBase.csv")
    return DataBase, link_list


def add_data(x):
    DataBase = x[0]
    nationality_list = []
    club_list = []
    height_list = []
    appearances = []
    position_list = []
    goal_list = []
    age_list = []
    for link in x[1]:
        url = requests.get(link)
        soup = BeautifulSoup(url.text, 'html.parser')
        nationality = soup.find('span', {"class" : "playerCountry"})
        if nationality is None:
            nationality_list.append(np.NaN)
        else:
            nationality_list.append(nationality.text)
        club = soup.find("span", {"class" : "long"})
        if club is None:
            club_list.append(np.NaN)
        else:
            clean = club.text.replace('\n', '')
            club_list.append(clean)
        info_height = soup.find_all("div", {"class" : "info"})
        for heights in info_height:
            hite = heights.text
            if "cm" in hite:
                height_list.append(hite)
        if len(height_list) < len(club_list):
            height_list.append(np.NaN)
        apps = soup.find("td", {"class" : "appearances"})
        if apps is None:
            appearances.append(np.NaN)
        else:
            apps = apps.text.replace('\n', '')
            appearances.append(apps)
        info_pos = soup.find_all("div", {"class" : "info"})
        positions = ["Goalkeeper", "Defender", "Midfielder", "Forward"]
        for line in info_pos:
            if line.text in positions:
                position_list.append(line.text)
                break
        goals = soup.find("td", {"class" : "goals"})
        if goals is None:
            goal_list.append(np.NaN)
        else:
            goals = goals.text.replace('\n', '')
            goal_list.append(goals)
        age = soup.find("span", {"class" : "info--light"})
        if age is None:
            age_list.append(np.NaN)
        else:
            age = age.text.replace('(','')
            age = age.replace(')','')
            age_list.append(age)
        print (f"{(x[1].index(link))+1} finished...")
    age_frame = pd.DataFrame(age_list, columns = ["Age"])
    goal_frame = pd.DataFrame(goal_list, columns = ["Goals"])
    position_frame = pd.DataFrame(position_list, columns = ["Position"])
    appearances_frame = pd.DataFrame(appearances, columns = ['Apps(Subs)'])
    height_frame = pd.DataFrame(height_list, columns = ["Height"])
    club_frame = pd.DataFrame(club_list, columns = ['Club'])
    nationality_frame = pd.DataFrame(nationality_list, columns = ['Nationality'])
    DataBase = x[0]
    DataBase = pd.concat([DataBase, age_frame, nationality_frame, club_frame, height_frame, appearances_frame, position_frame, goal_frame], axis = 1)
    print (DataBase)
    DataBase.to_csv("PlayerInfo.csv")

#initiate function
add_data(initial_scrape())
