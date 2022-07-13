import base64
import csv
import json
import os
import re
import shutil

import requests


url="http://39.105.102.68:8000/oldcare/event/"

#headers={'Authorization': 'token 52ee7d4c57686ca8d6884fa4c482a28'}




def insertevent(type, event_desc, event_location, oldid):
    payload = {
        'event_type': type,
        'event_location': event_location,
        'event_desc': event_desc,
        'oldperson': oldid
    }
    if oldid < 0 :
        payload = {'event_type': type,
        'event_location': event_location,
        'event_desc': event_desc}

    r = requests.post(url, data=payload)
    print(r.status_code)
    print(r.content)

def saveimage(p, head, type, writer):
    imagedir = './'
    writer.writerow([p.id, p.username, type])
    id = head + p.id
    if os.path.exists(os.path.join(imagedir, id)):
        shutil.rmtree(os.path.join(imagedir, id), True)
    os.mkdir(os.path.join(imagedir, id))
    i = 0
    for mystr in p.data.base64:
        result = re.search("data:image/(?P<ext>.*?);base64,(?P<data>.*)", mystr, re.DOTALL)
        if result:
            ext = result.groupdict().get("ext")
            data = result.groupdict().get("data")
        imgdata = base64.b64decode(data)
        file = open(os.path.join(imagedir, id) + '/' + str(i) + '.' + ext, 'wb')
        file.write(imgdata)
        file.close()
        i += 1

def getpeople():
    url = "http://39.105.102.68:8000/oldcare/face_inf/"
    csv_path = '../info/people_info.csv'
    imagedir = './'
    action_list = ['blink', 'open_mouth', 'emotion', 'rise_head', 'bow_head',
                   'look_left', 'look_right']
    r = requests.get(url)
    #print("content")
    #print(r.content)
    f = open(csv_path, 'w', encoding='utf-8')
    writer = csv.writer(f)
    writer.writerow(["id_card", "name", "type"])
    writer.writerow(["Unknown", "陌生人", ""])
    old = json.loads(r.text).get('old_people')
    volunteer = json.loads(r.text).get('volunteer')
    employee = json.loads(r.text).get('employee')
    print(volunteer)
    for p in old:
        writer.writerow(['1_' + str(p['id']), p['username'], "old_people"])
        #saveimage(p, '1_'+pid, "old_people", writer)
    for p in volunteer:
        writer.writerow(['2_' + str(p['id']), p['username'], "volunteer"])
        #saveimage(p, '2_'+pid, "volunteer", writer)
    for p in employee:
        writer.writerow(['3_' + str(p['id']), p['username'], "employee"])
        #saveimage(p, '3_'+pid, "employee", writer)

if __name__ == '__main__':
    getpeople()
