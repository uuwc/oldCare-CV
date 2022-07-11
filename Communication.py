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


if __name__ == '__main__':
    insertevent("摔倒检测", "呆哥家", "呆哥摔倒了", 1)
