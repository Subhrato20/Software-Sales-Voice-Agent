# server.py
import json
import os
import asyncio
import logging

from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from retell import Retell

from Agent.custom_types import (
    ConfigResponse,
    ResponseRequiredRequest,
)
from Agent.llm_with_func import LlmClient

# Load .env variables
load_dotenv(override=True)

# Logging setup
logging.basicConfig(
    #level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize Retell
RETELL_API_KEY = os.environ.get("RETELL_API_KEY")
if not RETELL_API_KEY:
    logger.error("RETELL_API_KEY not found in environment variables.")
    retell = None
else:
    retell = Retell(api_key=RETELL_API_KEY)


@app.post("/webhook")
async def handle_webhook(request: Request):
    if not retell:
        logger.error("Retell client not initialized.")
        return JSONResponse(status_code=500, content={"message": "Retell not configured"})
    try:
        body = await request.json()
        sig = request.headers.get("X-Retell-Signature")
        if not sig:
            return JSONResponse(status_code=401, content={"message": "Missing signature"})
        if not retell.verify(
            json.dumps(body, separators=(",", ":"), ensure_ascii=False),
            signature=sig
        ):
            return JSONResponse(status_code=401, content={"message": "Invalid signature"})
        evt = body.get("event")
        cid = body.get("data", {}).get("call_id", "N/A")
        logger.info(f"Webhook event '{evt}' for call_id '{cid}'")
        return JSONResponse(status_code=200, content={"received": True})
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"message": "Invalid JSON"})
    except Exception:
        logger.exception("Error in webhook")
        return JSONResponse(status_code=500, content={"message": "Internal Server Error"})


@app.websocket("/llm-websocket/{call_id}")
async def websocket_handler(websocket: WebSocket, call_id: str):
    await websocket.accept()
    logger.info(f"WebSocket connected for call_id: {call_id}")

    llm = LlmClient()

    # send config
    cfg = ConfigResponse(config={"auto_reconnect": True, "call_details": True}, response_id=0)
    await websocket.send_json(cfg.dict())

    # send ready event
    first = llm.draft_begin_message()
    await websocket.send_json(first.dict())

    try:
        while True:
            req = await websocket.receive_json()
            itype = req.get("interaction_type")

            if itype == "ping_pong":
                await websocket.send_json({"response_type": "ping_pong", "timestamp": req.get("timestamp")})
                continue

            if itype in ("call_details", "update_only"):
                logger.info(f"{itype} for {call_id}: {req}")
                continue

            if itype in ("response_required", "reminder_required"):
                rid = req.get("response_id")
                logger.info(f"Processing {itype} (response_id={rid}) for {call_id}")

                request_obj = ResponseRequiredRequest(
                    interaction_type=itype,
                    response_id=rid,
                    transcript=req.get("transcript", []),
                )

                async for chunk in llm.draft_response(request_obj):
                    if websocket.client_state.CONNECTED:
                        await websocket.send_json(chunk.dict())
                    else:
                        break
            else:
                logger.warning(f"Unknown interaction type: {itype}")

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnect for call_id: {call_id}")
    except Exception:
        logger.exception(f"Error in WebSocket for call_id: {call_id}")
        if websocket.client_state.CONNECTED:
            await websocket.close(code=1011, reason="Internal server error")
    finally:
        logger.info(f"WebSocket closed for call_id: {call_id}")
