from googleapiclient.discovery import build
api_key = ""

def fetch_trending_dog_shorts(api_key):
    youtube = build("youtube", "v3", developerKey=api_key)
    req = youtube.search().list(
        part="snippet",
        maxResults=10,
        q="dog #shorts",
        type="video",
        order="date"  # most recent
    )
    res = req.execute()
    return [
        {
            "title": item["snippet"]["title"],
            "videoId": item["id"]["videoId"],
            "publishedAt": item["snippet"]["publishedAt"]
        }
        for item in res["items"]
    ]
if __name__=="__main__":
    result=fetch_trending_dog_shorts(api_key)
    print(result)
