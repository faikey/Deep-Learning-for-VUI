from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import csv

driver = webdriver.Chrome("/Users/hanxu/Downloads/chromedriver")
#driver.fullscreen_window()
driver.get('https://developer.amazon.com/alexa/console/ask/test/amzn1.ask.skill.b78c5496-3dda-4085-ab6f-c51dfe3c785c/development/en_US/')

username = ""
password = ""

elem_email=driver.find_element_by_id("ap_email")
elem_email.send_keys(username)
elem_ps = driver.find_element_by_id("ap_password")
elem_ps.send_keys(password)
elem_si = driver.find_element_by_id("signInSubmit")
elem_si.click()


datas = ['name','question','response']
with open('social_command_response2.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(datas)

    with open("Social.csv") as skill:
        reader = csv.reader(skill)
        firstline = True
        skillname = ""
        global opencommand
        for r in reader:
            if firstline:
                firstline = False
                continue

            if skillname != str(r[0]):
                opencommand = str(r[1])
                if int(r[2]) >=5:
                    j = str(r[0])
                    skillname = j
                    row = []
                    row.append(j)        
                    command = str(r[1])
                    row.append(command)

                    elem = driver.find_element_by_class_name("askt-utterance__input")
                    elem.send_keys(command)
    #enter = driver.find_element_by_class_name("askt-utterance")
    #enter.send_keys(Keys.ENTER)
                    elem.send_keys(Keys.ENTER)

                    try:
                        content = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".askt-dialog__message.askt-dialog__message--active-response")))
                    #print(content.text)
                        if not content.text:
                            driver.get('https://developer.amazon.com/alexa/console/ask/test/amzn1.ask.skill.b78c5496-3dda-4085-ab6f-c51dfe3c785c/development/en_US/')
                            time.sleep(3)
                            elem = driver.find_element_by_class_name("askt-utterance__input")
                            elem.send_keys(command)
                            elem.send_keys(Keys.ENTER)
                            try:
                                content = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".askt-dialog__message.askt-dialog__message--response")))
                                time.sleep(3)
                                first = True
                                for i in range(len(content)):
                                    if first:
                                        row.append(i.text)
                                        first = False
                                    else:
                                        row[2] += i.text
                            except:
                                pass
                            try:
                                content_new = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".askt-dialog__message.askt-dialog__message--active-response")))
                                row[2] += content_new.text
                            except:
                                pass
                        else:
                            row.append(content.text)   
        #content = driver.find_element_by_css_selector(".askt-dialog__message.askt-dialog__message--active-response")
                    except:
                        row.append("")
                        driver.get('https://developer.amazon.com/alexa/console/ask/test/amzn1.ask.skill.b78c5496-3dda-4085-ab6f-c51dfe3c785c/development/en_US/')
                        time.sleep(3)
                        elem = driver.find_element_by_class_name("askt-utterance__input")
                        elem.send_keys(command)
                        elem.send_keys(Keys.ENTER)
                        try:
                            content = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".askt-dialog__message.askt-dialog__message--response")))
                            time.sleep(3)
                            for i in range(len(content)):
                                row[2] += i.text
                        except:
                            pass
                        try:
                            content_new = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".askt-dialog__message.askt-dialog__message--active-response")))
                            row[2] += content_new.text
                        except:
                            pass
                        
                         
    #print(content)
                    time.sleep(3)
                    writer.writerow(row)
                    if len(row) != 3:
                        row.append("")

                    if row[2] != "" and "Do you want to open it?" in str(row[2]):
                        row = []
                        row.append(j)
                        command = "Yes"
                        row.append(command)

                        elem = driver.find_element_by_class_name("askt-utterance__input")
                        elem.send_keys(command)
    #enter = driver.find_element_by_class_name("askt-utterance")
    #enter.send_keys(Keys.ENTER)
                        elem.send_keys(Keys.ENTER)
                        try:
                            content = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".askt-dialog__message.askt-dialog__message--active-response")))
                    #print(content.text)
                            if not content.text:
                                driver.get('https://developer.amazon.com/alexa/console/ask/test/amzn1.ask.skill.b78c5496-3dda-4085-ab6f-c51dfe3c785c/development/en_US/')
                                time.sleep(3)
                                elem = driver.find_element_by_class_name("askt-utterance__input")
                                elem.send_keys(command)
                                elem.send_keys(Keys.ENTER)
                                try:
                                    content = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".askt-dialog__message.askt-dialog__message--response")))
                                    time.sleep(3)
                                    first = True
                                    for i in range(len(content)):
                                        if first:
                                            row.append(i.text)
                                            first = False
                                        else:
                                            row[2] += i.text
                                except:
                                    pass
                                try:
                                    content_new = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".askt-dialog__message.askt-dialog__message--active-response")))
                                    row[2] += content_new.text
                                except:
                                    pass
                            else:
                                row.append(content.text)   
                        except:
                            row.append("")
                            driver.get('https://developer.amazon.com/alexa/console/ask/test/amzn1.ask.skill.b78c5496-3dda-4085-ab6f-c51dfe3c785c/development/en_US/')
                            time.sleep(3)
                            elem = driver.find_element_by_class_name("askt-utterance__input")
                            elem.send_keys(command)
                            elem.send_keys(Keys.ENTER)
                            try:
                                content = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".askt-dialog__message.askt-dialog__message--response")))
                                time.sleep(3)
                                for i in range(len(content)):
                                    row[2] += i.text
                            except:
                                pass
                            try:
                                content_new = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".askt-dialog__message.askt-dialog__message--active-response")))
                                row[2] += content_new.text
                            except:
                                pass
    #print(content)
                        time.sleep(3)
                        writer.writerow(row)

                        row = []
                        row.append(j)        
                        command = str(r[1])
                        row.append(command)

                        elem = driver.find_element_by_class_name("askt-utterance__input")
                        elem.send_keys(command)
    #enter = driver.find_element_by_class_name("askt-utterance")
    #enter.send_keys(Keys.ENTER)
                        elem.send_keys(Keys.ENTER)

                        try:
                            content = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".askt-dialog__message.askt-dialog__message--active-response")))
                    #print(content.text)
                            if not content.text:
                                driver.get('https://developer.amazon.com/alexa/console/ask/test/amzn1.ask.skill.b78c5496-3dda-4085-ab6f-c51dfe3c785c/development/en_US/')
                                time.sleep(3)
                                elem = driver.find_element_by_class_name("askt-utterance__input")
                                elem.send_keys(command)
                                elem.send_keys(Keys.ENTER)
                                try:
                                    content = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".askt-dialog__message.askt-dialog__message--response")))
                                    time.sleep(3)
                                    first = True
                                    for i in range(len(content)):
                                        if first:
                                            row.append(i.text)
                                            first = False
                                        else:
                                            row[2] += i.text
                                except:
                                    pass
                                try:
                                    content_new = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".askt-dialog__message.askt-dialog__message--active-response")))
                                    row[2] += content_new.text
                                except:
                                    pass
                            else:
                                row.append(content.text)   
                        except:
                            row.append("")
                            driver.get('https://developer.amazon.com/alexa/console/ask/test/amzn1.ask.skill.b78c5496-3dda-4085-ab6f-c51dfe3c785c/development/en_US/')
                            time.sleep(3)
                            elem = driver.find_element_by_class_name("askt-utterance__input")
                            elem.send_keys(command)
                            elem.send_keys(Keys.ENTER)
                            try:
                                content = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".askt-dialog__message.askt-dialog__message--response")))
                                time.sleep(3)
                                for i in range(len(content)):
                                    row[2] += i.text
                            except:
                                pass
                            try:
                                content_new = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".askt-dialog__message.askt-dialog__message--active-response")))
                                row[2] += content_new.text
                            except:
                                pass
    #print(content)
                        time.sleep(3)
                        writer.writerow(row)

                    row = []
                    row.append(j)        
                    command = "quit"
                    row.append(command)

                    elem = driver.find_element_by_class_name("askt-utterance__input")
                    elem.send_keys(command)
    #enter = driver.find_element_by_class_name("askt-utterance")
    #enter.send_keys(Keys.ENTER)
                    elem.send_keys(Keys.ENTER)
                    try:
                        content = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".askt-dialog__message.askt-dialog__message--active-response")))
                    #print(content.text)
                        row.append(content.text)
    #content = driver.find_element_by_css_selector(".askt-dialog__message.askt-dialog__message--active-response")
                    except:
                        row.append("")
                    
    #print(content)
                    time.sleep(3)
                    writer.writerow(row)
                else: 
                    break

            else:       
                command = str(r[1])
                skillname = str(r[0])

                if "Alexa" not in command:
                    j = str(r[0])
                    row = []
                    row.append(j) 
                    row.append(opencommand)
                    elem = driver.find_element_by_class_name("askt-utterance__input")
                    elem.send_keys(opencommand)
    #enter = driver.find_element_by_class_name("askt-utterance")
    #enter.send_keys(Keys.ENTER)
                    elem.send_keys(Keys.ENTER)

                    try:
                        content = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".askt-dialog__message.askt-dialog__message--active-response")))
                    #print(content.text)
                        if not content.text:
                            driver.get('https://developer.amazon.com/alexa/console/ask/test/amzn1.ask.skill.b78c5496-3dda-4085-ab6f-c51dfe3c785c/development/en_US/')
                            time.sleep(3)
                            elem = driver.find_element_by_class_name("askt-utterance__input")
                            elem.send_keys(command)
                            elem.send_keys(Keys.ENTER)
                            try:
                                content = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".askt-dialog__message.askt-dialog__message--response")))
                                time.sleep(3)
                                first = True
                                for i in range(len(content)):
                                    if first:
                                        row.append(i.text)
                                        first = False
                                    else:
                                        row[2] += i.text
                            except:
                                pass
                            try:
                                content_new = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".askt-dialog__message.askt-dialog__message--active-response")))
                                row[2] += content_new.text
                            except:
                                pass
                        else:
                            row.append(content.text)    
                    except:
                        row.append("")
                        driver.get('https://developer.amazon.com/alexa/console/ask/test/amzn1.ask.skill.b78c5496-3dda-4085-ab6f-c51dfe3c785c/development/en_US/')
                        time.sleep(3)
                        elem = driver.find_element_by_class_name("askt-utterance__input")
                        elem.send_keys(command)
                        elem.send_keys(Keys.ENTER)
                        try:
                            content = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".askt-dialog__message.askt-dialog__message--response")))
                            time.sleep(3)
                            for i in range(len(content)):
                                row[2] += i.text
                        except:
                            pass
                        try:
                            content_new = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".askt-dialog__message.askt-dialog__message--active-response")))
                            row[2] += content_new.text
                        except:
                            pass
    #print(content)
                    time.sleep(3)
                    writer.writerow(row)
                    if len(row)!=3:
                        row.append("")

                    if row[2] != "" and "Do you want to open it?" in str(row[2]):
                        row = []
                        row.append(j)
                        command = "Yes"
                        row.append(command)

                        elem = driver.find_element_by_class_name("askt-utterance__input")
                        elem.send_keys(command)
    #enter = driver.find_element_by_class_name("askt-utterance")
    #enter.send_keys(Keys.ENTER)
                        elem.send_keys(Keys.ENTER)
                        try:
                            content = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".askt-dialog__message.askt-dialog__message--active-response")))
                    #print(content.text)
                            if not content.text:
                                driver.get('https://developer.amazon.com/alexa/console/ask/test/amzn1.ask.skill.b78c5496-3dda-4085-ab6f-c51dfe3c785c/development/en_US/')
                                time.sleep(3)
                                elem = driver.find_element_by_class_name("askt-utterance__input")
                                elem.send_keys(command)
                                elem.send_keys(Keys.ENTER)
                                try:
                                    content = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".askt-dialog__message.askt-dialog__message--response")))
                                    time.sleep(3)
                                    first = True
                                    for i in range(len(content)):
                                        if first:
                                            row.append(i.text)
                                            first = False
                                        else:
                                            row[2] += i.text
                                except:
                                    pass
                                try:
                                    content_new = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".askt-dialog__message.askt-dialog__message--active-response")))
                                    row[2] += content_new.text
                                except:
                                    pass
                            else:
                                row.append(content.text)   
                        except:
                            row.append("")
                            driver.get('https://developer.amazon.com/alexa/console/ask/test/amzn1.ask.skill.b78c5496-3dda-4085-ab6f-c51dfe3c785c/development/en_US/')
                            time.sleep(3)
                            elem = driver.find_element_by_class_name("askt-utterance__input")
                            elem.send_keys(command)
                            elem.send_keys(Keys.ENTER)
                            try:
                                content = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".askt-dialog__message.askt-dialog__message--response")))
                                time.sleep(3)
                                for i in range(len(content)):
                                    row[2] += i.text
                            except:
                                pass
                            try:
                                content_new = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".askt-dialog__message.askt-dialog__message--active-response")))
                                row[2] += content_new.text
                            except:
                                pass

                    row = []
                    row.append(j) 
                    row.append(command)
                    elem = driver.find_element_by_class_name("askt-utterance__input")
                    elem.send_keys(command)
    #enter = driver.find_element_by_class_name("askt-utterance")
    #enter.send_keys(Keys.ENTER)
                    elem.send_keys(Keys.ENTER)

                    try:
                        content = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".askt-dialog__message.askt-dialog__message--active-response")))
                    #print(content.text)
                        if not content.text:
                            driver.get('https://developer.amazon.com/alexa/console/ask/test/amzn1.ask.skill.b78c5496-3dda-4085-ab6f-c51dfe3c785c/development/en_US/')
                            time.sleep(3)
                            elem = driver.find_element_by_class_name("askt-utterance__input")
                            elem.send_keys(command)
                            elem.send_keys(Keys.ENTER)
                            try:
                                content = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".askt-dialog__message.askt-dialog__message--response")))
                                time.sleep(3)
                                first = True
                                for i in range(len(content)):
                                    if first:
                                        row.append(i.text)
                                        first = False
                                    else:
                                        row[2] += i.text
                            except:
                                pass
                            try:
                                content_new = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".askt-dialog__message.askt-dialog__message--active-response")))
                                row[2] += content_new.text
                            except:
                                pass
                        else:
                            row.append(content.text)    
                    except:
                        row.append("")
                        driver.get('https://developer.amazon.com/alexa/console/ask/test/amzn1.ask.skill.b78c5496-3dda-4085-ab6f-c51dfe3c785c/development/en_US/')
                        time.sleep(3)
                        elem = driver.find_element_by_class_name("askt-utterance__input")
                        elem.send_keys(command)
                        elem.send_keys(Keys.ENTER)
                        try:
                            content = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".askt-dialog__message.askt-dialog__message--response")))
                            time.sleep(3)
                            for i in range(len(content)):
                                row[2] += i.text
                        except:
                            pass
                        try:
                            content_new = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".askt-dialog__message.askt-dialog__message--active-response")))
                            row[2] += content_new.text
                        except:
                            pass
    #print(content)
                    time.sleep(3)
                    writer.writerow(row)

                    row = []
                    row.append(j)        
                    command = "quit"
                    row.append(command)

                    elem = driver.find_element_by_class_name("askt-utterance__input")
                    elem.send_keys(command)
    #enter = driver.find_element_by_class_name("askt-utterance")
    #enter.send_keys(Keys.ENTER)
                    elem.send_keys(Keys.ENTER)
                    try:
                        content = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".askt-dialog__message.askt-dialog__message--active-response")))
                    #print(content.text)
                        row.append(content.text)
    #content = driver.find_element_by_css_selector(".askt-dialog__message.askt-dialog__message--active-response")
                    except:
                        row.append("")
                    
    #print(content)
                    time.sleep(3)
                    writer.writerow(row)

                else:
                    row = []
                    row.append(j) 
                    row.append(command)
                    elem = driver.find_element_by_class_name("askt-utterance__input")
                    elem.send_keys(command)
    #enter = driver.find_element_by_class_name("askt-utterance")
    #enter.send_keys(Keys.ENTER)
                    elem.send_keys(Keys.ENTER)

                    try:
                        content = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".askt-dialog__message.askt-dialog__message--active-response")))
                    #print(content.text)
                        if not content.text:
                            driver.get('https://developer.amazon.com/alexa/console/ask/test/amzn1.ask.skill.b78c5496-3dda-4085-ab6f-c51dfe3c785c/development/en_US/')
                            time.sleep(3)
                            elem = driver.find_element_by_class_name("askt-utterance__input")
                            elem.send_keys(command)
                            elem.send_keys(Keys.ENTER)
                            try:
                                content = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".askt-dialog__message.askt-dialog__message--response")))
                                time.sleep(3)
                                first = True
                                for i in range(len(content)):
                                    if first:
                                        row.append(i.text)
                                        first = False
                                    else:
                                        row[2] += i.text
                            except:
                                pass
                            try:
                                content_new = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".askt-dialog__message.askt-dialog__message--active-response")))
                                row[2] += content_new.text
                            except:
                                pass
                        else:
                            row.append(content.text)   
                    except:
                        row.append("")
                        driver.get('https://developer.amazon.com/alexa/console/ask/test/amzn1.ask.skill.b78c5496-3dda-4085-ab6f-c51dfe3c785c/development/en_US/')
                        time.sleep(3)
                        elem = driver.find_element_by_class_name("askt-utterance__input")
                        elem.send_keys(command)
                        elem.send_keys(Keys.ENTER)
                        try:
                            content = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".askt-dialog__message.askt-dialog__message--response")))
                            time.sleep(3)
                            for i in range(len(content)):
                                row[2] += i.text
                        except:
                            pass
                        try:
                            content_new = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".askt-dialog__message.askt-dialog__message--active-response")))
                            row[2] += content_new.text
                        except:
                            pass
    #print(content)
                    time.sleep(3)
                    writer.writerow(row)
                    if len(row)!=3:
                        row.append("")

                    if row[2] != "" and "Do you want to open it?" in str(row[2]):
                        row = []
                        row.append(j)
                        command = "Yes"
                        row.append(command)

                        elem = driver.find_element_by_class_name("askt-utterance__input")
                        elem.send_keys(command)
    #enter = driver.find_element_by_class_name("askt-utterance")
    #enter.send_keys(Keys.ENTER)
                        elem.send_keys(Keys.ENTER)
                        try:
                            content = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".askt-dialog__message.askt-dialog__message--active-response")))
                    #print(content.text)
                            if not content.text:
                                driver.get('https://developer.amazon.com/alexa/console/ask/test/amzn1.ask.skill.b78c5496-3dda-4085-ab6f-c51dfe3c785c/development/en_US/')
                                time.sleep(3)
                                elem = driver.find_element_by_class_name("askt-utterance__input")
                                elem.send_keys(command)
                                elem.send_keys(Keys.ENTER)
                                try:
                                    content = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".askt-dialog__message.askt-dialog__message--response")))
                                    time.sleep(3)
                                    first = True
                                    for i in range(len(content)):
                                        if first:
                                            row.append(i.text)
                                            first = False
                                        else:
                                            row[2] += i.text
                                except:
                                    pass
                                try:
                                    content_new = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".askt-dialog__message.askt-dialog__message--active-response")))
                                    row[2] += content_new.text
                                except:
                                    pass
                            else:
                                row.append(content.text)   
                        except:
                            row.append("")
                            driver.get('https://developer.amazon.com/alexa/console/ask/test/amzn1.ask.skill.b78c5496-3dda-4085-ab6f-c51dfe3c785c/development/en_US/')
                            time.sleep(3)
                            elem = driver.find_element_by_class_name("askt-utterance__input")
                            elem.send_keys(command)
                            elem.send_keys(Keys.ENTER)
                            try:
                                content = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".askt-dialog__message.askt-dialog__message--response")))
                                time.sleep(3)
                                for i in range(len(content)):
                                    row[2] += i.text
                            except:
                                pass
                            try:
                                content_new = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".askt-dialog__message.askt-dialog__message--active-response")))
                                row[2] += content_new.text
                            except:
                                pass
    #print(content)
                        time.sleep(3)
                        writer.writerow(row)

                        row = []
                        row.append(j)        
                        command = str(r[1])
                        row.append(command)

                        elem = driver.find_element_by_class_name("askt-utterance__input")
                        elem.send_keys(command)
    #enter = driver.find_element_by_class_name("askt-utterance")
    #enter.send_keys(Keys.ENTER)
                        elem.send_keys(Keys.ENTER)

                        try:
                            content = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".askt-dialog__message.askt-dialog__message--active-response")))
                    #print(content.text)
                            if not content.text:
                                driver.get('https://developer.amazon.com/alexa/console/ask/test/amzn1.ask.skill.b78c5496-3dda-4085-ab6f-c51dfe3c785c/development/en_US/')
                                time.sleep(3)
                                elem = driver.find_element_by_class_name("askt-utterance__input")
                                elem.send_keys(command)
                                elem.send_keys(Keys.ENTER)
                                try:
                                    content = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".askt-dialog__message.askt-dialog__message--response")))
                                    time.sleep(3)
                                    first = True
                                    for i in range(len(content)):
                                        if first:
                                            row.append(i.text)
                                            first = False
                                        else:
                                            row[2] += i.text
                                except:
                                    pass
                                try:
                                    content_new = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".askt-dialog__message.askt-dialog__message--active-response")))
                                    row[2] += content_new.text
                                except:
                                    pass
                            else:
                                row.append(content.text)   
                        except:
                            row.append("")
                            driver.get('https://developer.amazon.com/alexa/console/ask/test/amzn1.ask.skill.b78c5496-3dda-4085-ab6f-c51dfe3c785c/development/en_US/')
                            time.sleep(3)
                            elem = driver.find_element_by_class_name("askt-utterance__input")
                            elem.send_keys(command)
                            elem.send_keys(Keys.ENTER)
                            try:
                                content = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".askt-dialog__message.askt-dialog__message--response")))
                                time.sleep(3)
                                for i in range(len(content)):
                                    row[2] += i.text
                            except:
                                pass
                            try:
                                content_new = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".askt-dialog__message.askt-dialog__message--active-response")))
                                row[2] += content_new.text
                            except:
                                pass
    #print(content)
                        time.sleep(3)
                        writer.writerow(row)

                    row = []
                    row.append(j)        
                    command = "quit"
                    row.append(command)

                    elem = driver.find_element_by_class_name("askt-utterance__input")
                    elem.send_keys(command)
    #enter = driver.find_element_by_class_name("askt-utterance")
    #enter.send_keys(Keys.ENTER)
                    elem.send_keys(Keys.ENTER)
                    try:
                        content = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".askt-dialog__message.askt-dialog__message--active-response")))
                    #print(content.text)
                        row.append(content.text)
    #content = driver.find_element_by_css_selector(".askt-dialog__message.askt-dialog__message--active-response")
                    except:
                        row.append("")
                    
    #print(content)
                    time.sleep(3)
                    writer.writerow(row)
                        
                    
    #print(content)
                    time.sleep(3)
                    writer.writerow(row)









