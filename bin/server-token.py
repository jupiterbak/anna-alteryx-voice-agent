import os
from livekit import api
from flask import Flask

os.environ['LIVEKIT_API_KEY'] = "devkey"
os.environ['LIVEKIT_API_SECRET'] = "secret"

app = Flask(__name__)

@app.route('/getToken')
def getToken():
    token = api.AccessToken(os.getenv('LIVEKIT_API_KEY'), os.getenv('LIVEKIT_API_SECRET')) \
        .with_identity("identity") \
        .with_name("my name") \
        .with_grants(api.VideoGrants(
            room_join=True,
            room="my-room",
        ))
    return token.to_jwt()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7880)