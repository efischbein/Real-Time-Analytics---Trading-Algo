import websocket
import json
import requests
import os
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import torch
from transformers import BertTokenizer, BertForSequenceClassification

def get_user_choice():
    valid_choices = ['nltk', 'bert', 'chatgpt']

    while True:
        user_choice = input("Enter your choice (NLTK, BERT, or ChatGPT): ").lower()

        if user_choice in valid_choices:
            return user_choice
        else:
            print("Invalid choice. Please try again.")

model_selection = get_user_choice()
print(f"Model Selected: {model_selection}")

# Specify the dollar amount you want to invest
dollar_amount = 1000

# Load environment variables from .env file
load_dotenv()

alpaca_api_key = os.getenv("ALPACA_API_KEY")
alpaca_secret_key = os.getenv("ALPACA_SECRET_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
base_url = "https://paper-api.alpaca.markets"

api = tradeapi.REST(key_id=alpaca_api_key, secret_key=alpaca_secret_key,  base_url=base_url)

# Function to calculate sentiment score
nltk.downloader.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'  # Model name
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

def get_sentiment_nltk(text):
    sentiment = sia.polarity_scores(text)

    compound_score = sentiment['compound']

    if compound_score >= 0.05:
        return 'BUY'
    elif compound_score <= -0.05:
        return 'SELL'
    else:
        return 'DO NOTHING'

def get_sentiment_bert(text):
    # Preprocess the news article
    tokens = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,  # Max input length supported by BERT
        padding='max_length',
        truncation=True,
        return_tensors='pt'  # Return PyTorch tensors
    )

    # Perform inference
    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)

    # Define the label mapping
    sentiment_labels = {0: "DO NOTHING", 1: "BUY", 2: "SELL"}

    # Map the predicted label to the corresponding sentiment
    sentiment = sentiment_labels[predictions.item()]

    return sentiment

def on_open(ws):
    print("Websocket connected!")

    # We now have to log in to the data source
    auth_msg = {
        "action": "auth",
        "key": alpaca_api_key,
        "secret": alpaca_secret_key
    }

    ws.send(json.dumps(auth_msg))  # Send auth data to ws, "log us in"

    # Subscribe to all news feeds
    subscribe_msg = {
        "action": "subscribe",
        "news": ["*"]  # ["TSLA"]
    }
    ws.send(json.dumps(subscribe_msg))  # Connecting us to the live data source of news

def on_message(ws, message):
    print("Message is " + message)
    # message is a STRING
    current_event = json.loads(message)[0]
    # "T": "n" newsEvent
    if current_event["T"] == "n":  # This is a news event

        # Make trades based on the sentiment
        ticker_symbol = current_event["symbols"][0]
        print(f"Ticker Symbol: {ticker_symbol}")

        try:
            latest_price = api.get_snapshot(symbol=ticker_symbol).latest_quote.ap

            # # Calculate the number of shares based on the dollar amount and latest price
            shares = int(dollar_amount / latest_price)
            print(f"Buy {shares} shares.")
            print(f"Latest Price: {latest_price}")
        except Exception as e:
            # default to one share
            shares = 1
            print(f"Error getting price: {str(e)}")


        if model_selection == 'chatgpt':

            # Ask ChatGPT its thoughts on the headline
            api_request_body = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "Only respond with BUY, SELL, or DO NOTHING based on the impact of the headline."},  # How ChatGPT should talk to us
                    {"role": "user", "content": "Given the headline '" + current_event["headline"] + "', give me a BUY, SELL, or DO NOTHING recommendation detailing the impact of this headline with no other text."}
                ]
            }

            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": "Bearer " + openai_api_key,
                    "Content-Type": "application/json"
                },
                json=api_request_body
            )
            data = response.json()
            print(data)
            sentiment = data["choices"][0]["message"]["content"].replace('.','')
            print(f"Sentiment for article: {sentiment}")

        elif model_selection == 'nltk':
            # calculate article sentimement
            sentiment = get_sentiment_nltk(current_event["headline"])
            print(f"Sentiment for article: {sentiment}")

        elif model_selection == 'bert':
            # calculate article sentimement score
            sentiment = get_sentiment_bert(current_event["headline"])
            print(f"Sentiment for article: {sentiment}")


        if sentiment == 'BUY': # BUY STOCK
        # Buy stock
            order = api.submit_order(
                symbol=ticker_symbol,
                qty=shares,
                side="buy",
                type="market",
                time_in_force="day"  # day ends, it won't trade.
                )
            print('Order Submitted')
        elif sentiment == 'SELL':  #SELL ALL STOCK
            # Sell stock
            order = api.submit_order(
                symbol=ticker_symbol,
                qty=0,
                side="sell",
                type="market",
                time_in_force="day"  # day ends, it won't trade.
            )
            print('Position Closed')

web_socket_url = r'wss://stream.data.alpaca.markets/v1beta1/news'

ws = websocket.WebSocketApp(web_socket_url, on_open=on_open, on_message=on_message)
ws.run_forever()
