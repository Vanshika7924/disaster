import requests

API_KEY = "bc137f28a31ed966558bc3bc8ec88bc4"
BASE_URL = "https://api.openweathermap.org/data/2.5/weather?q=Bhopal&appid=bc137f28a31ed966558bc3bc8ec88bc4"

def fetch_disaster_news():
    try:
        params = {
            "q": "flood OR earthquake OR cyclone OR fire OR disaster",
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 5,
            "apiKey": API_KEY
        }

        response = requests.get(BASE_URL, params=params, timeout=10)
        data = response.json()

        news_texts = []

        for article in data.get("articles", []):
            title = article.get("title") or ""
            description = article.get("description") or ""
            raw_text = (title + " " + description).strip()

            if raw_text:
                news_texts.append(raw_text)

        return news_texts

    except Exception as e:
        print("News fetch error:", e)
        return []