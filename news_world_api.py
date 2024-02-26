import json
import urllib.request
import datetime
apikey = 
start_date = datetime.date(2023, 6, 8)
end_date = datetime.date(2023, 6, 8)
delta = datetime.timedelta(days=1)
f = open('news_world_api.csv', 'a')
user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
headers={'User-Agent':user_agent,} 
while (start_date <= end_date):
    url = f"https://api.worldnewsapi.com/search-news?text=sber&earliest-publish-date=2023-06-23&api-key={apikey}"
    request=urllib.request.Request(url,None,headers)
    response = urllib.request.urlopen(request)
    data = json.loads(response.read().decode("utf-8"))
    f.write(str(start_date) + ';' + str(data['available']) + '\n')
    print(str(start_date) + ';' + str(data['available']))
    start_date += delta
print('end')
f.close()