import os
import json
import base64
import asyncio
import websockets
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.websockets import WebSocketDisconnect
from twilio.twiml.voice_response import VoiceResponse, Connect, Say, Stream

app = FastAPI()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
PORT = int(os.environ.get('PORT', 8080))

SYSTEM_MESSAGE = "You are a helpful AI voice assistant. Be concise and friendly."
VOICE = 'alloy'

LOG_EVENT_TYPES = [
    'response.content.done', 'rate_limits.updated', 'response.done',
    'input_audio_buffer.committed', 'input_audio_buffer.speech_stopped',
    'input_audio_buffer.speech_started', 'session.created'
]

@app.get("/")
async def root():
    return {"message": "Voice Agent is running"}

@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    response = VoiceResponse()
    host = request.url.hostname
    connect = Connect()
    connect.stream(url=f'wss://{host}/media-stream')
    response.append(connect)
    return HTMLResponse(content=str(response), media_type="application/xml")

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    await websocket.accept()
    async with websockets.connect(
        'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01',
        additional_headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1"
        }
    ) as openai_ws:
        await send_session_update(openai_ws)
        stream_sid = None

        async def receive_from_twilio():
            nonlocal stream_sid
            try:
                async for message in websocket.iter_text():
                    data = json.loads(message)
                    if data['event'] == 'media' and openai_ws.open:
                        audio_append = {
                            "type": "input_audio_buffer.append",
                            "audio": data['media']['payload']
                        }
                        await openai_ws.send(json.dumps(audio_append))
                    elif data['event'] == 'start':
                        stream_sid = data['start']['streamSid']
            except WebSocketDisconnect:
                if openai_ws.open:
                    await openai_ws.close()

        async def send_to_twilio():
            nonlocal stream_sid
            try:
                async for openai_message in openai_ws:
                    response = json.loads(openai_message)
                    if response['type'] in LOG_EVENT_TYPES:
                        print(f"Event: {response['type']}", flush=True)
                    if response['type'] == 'session.updated':
                        print("Session updated successfully", flush=True)
                    if response['type'] == 'response.audio.delta' and response.get('delta'):
                        audio_delta = {
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {
                                "payload": base64.b64encode(
                                    base64.b64decode(response['delta'])
                                ).decode('utf-8')
                            }
                        }
                        await websocket.send_json(audio_delta)
            except Exception as e:
                print(f"Error: {e}", flush=True)

        await asyncio.gather(receive_from_twilio(), send_to_twilio())

async def send_session_update(openai_ws):
    session_update = {
        "type": "session.update",
        "session": {
            "turn_detection": {"type": "server_vad"},
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "voice": VOICE,
            "instructions": SYSTEM_MESSAGE,
            "modalities": ["text", "audio"],
            "temperature": 0.8,
        }
    }
    await openai_ws.send(json.dumps(session_update))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
