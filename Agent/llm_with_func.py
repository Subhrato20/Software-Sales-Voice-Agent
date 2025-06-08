import logging
import os
import json
import asyncio
from typing import List

from openai import AsyncOpenAI
from .custom_types import ResponseRequiredRequest, ResponseResponse, Utterance
from RAG.milvus_search import RAGHandler

# — Configure module-wide logging —
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

BEGIN_SENTENCE = (
    "Hello! Thanks for your interest in our software services. "
    "I'm here to help answer any questions you have. How can I assist you today?"
)

AGENT_PROMPT = (
    "Task: You are an AI Software Sales Agent. Your primary goal is to understand "
    "prospective customers' needs, provide them with relevant information about our software, "
    "and guide them towards a solution that fits their requirements.\n\n"
    "Whenever the user asks a question, you MUST call the `knowledge_base_search` function "
    "to retrieve the answer. If the knowledge base has no answer, respond exactly: \"I don’t know.\"\n\n"
    "When the user asks about features, pricing, technical specs, comparisons, or FAQs, "
    "use the `knowledge_base_search` function to fetch the information.\n\n"
    "CRITICAL:\n"
    "- Tone: Professional, friendly, confident.\n"
    "- Conciseness: Keep answers focused and to the point.\n"
    "- Accuracy: Never invent—always rely on the knowledge base.\n"
    "- Closing: If the user indicates they’re done, call the `end_call` function.\n"
)


class LlmClient:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set in the environment")
        self.client = AsyncOpenAI(
            api_key=api_key,
        )
        self.model_name = os.getenv(
            "GEMINI_MODEL", "gpt-4o"
        )
        self.rag_handler = RAGHandler()

    def draft_begin_message(self) -> ResponseResponse:
        return ResponseResponse(
            response_id=0,
            content=BEGIN_SENTENCE,
            content_complete=True,
            end_call=False,
        )

    def convert_transcript_to_openai_messages(
        self, transcript: List[Utterance]
    ) -> List[dict]:
        msgs = []
        for ut in transcript:
            role = "assistant" if ut.role == "agent" else "user"
            msgs.append({"role": role, "content": ut.content})
        return msgs

    def prepare_prompt_messages(
        self, request: ResponseRequiredRequest
    ) -> List[dict]:
        system = {
            "role": "system",
            "content": "You are in a voice call. Respond based on the transcript and rules:\n\n" + AGENT_PROMPT
        }
        user_msgs = self.convert_transcript_to_openai_messages(request.transcript)
        if request.interaction_type == "reminder_required":
            user_msgs.append({
                "role": "user",
                "content": "(User idle—please re-engage.)"
            })
        return [system] + user_msgs

    def prepare_functions(self) -> List[dict]:
        return [
            {
                "name": "knowledge_base_search",
                "description": (
                    "Search the software knowledge base for features, pricing, "
                    "technical details, use cases, comparisons, or FAQs."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The user's question or topic to search for."
                        }
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "end_call",
                "description": "Ends the call when the user is finished.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Final polite message."
                        }
                    },
                    "required": ["message"],
                    "additionalProperties": False,
                },
            },
        ]

    async def handle_tool_call_response(
        self,
        request: ResponseRequiredRequest,
        function_name: str,
        original_query: str,
        tool_result: str,
    ):
        """
        Takes the raw tool_result (the <page>…</page> blobs) and asks the LLM
        to turn it into a concise, friendly answer to the user's original_query.
        """
        try:
            # 1) Build a fresh prompt that tells the model to reformat the KB output.
            formatting_msgs = [
                {
                    "role": "system",
                    "content": (
                        "You are a professional, friendly AI assistant. "
                        "The user asked: “" + original_query + "”.\n\n"
                        "Below is the raw information retrieved from the knowledge base.\n"
                        "Please condense it into a concise, accurate, and helpful reply. and remove **"
                    ),
                },
                {
                    "role": "user",
                    "content": tool_result,
                },
            ]

            logger.debug("Formatting tool result into final answer…")
            stream = await self.client.chat.completions.create(
                model=self.model_name,
                messages=formatting_msgs,
                stream=True,
            )

            full = ""
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    full += delta.content
                    yield ResponseResponse(
                        response_id=request.response_id,
                        content=delta.content,
                        content_complete=False,
                        end_call=False,
                    )

            # signal end of turn
            if full:
                yield ResponseResponse(
                    response_id=request.response_id,
                    content="",
                    content_complete=True,
                    end_call=False,
                )

        except Exception:
            logger.exception("Error formatting the answer")
            yield ResponseResponse(
                response_id=request.response_id,
                content="Sorry, something went wrong formatting the answer.",
                content_complete=True,
                end_call=False,
            )


    async def draft_response(self, request: ResponseRequiredRequest):
        prompt_msgs = self.prepare_prompt_messages(request)
        logger.debug("Prompt messages:\n%s", json.dumps(prompt_msgs, indent=2))

        try:
            stream = await self.client.chat.completions.create(
                model=self.model_name,
                messages=prompt_msgs,
                stream=True,
                functions=self.prepare_functions(),
                function_call="auto",
            )

            func_name = None
            func_args = ""

            async for chunk in stream:
                logger.debug("↳ LLM chunk: %s", chunk)
                delta = chunk.choices[0].delta

                # 1) Plain-text response
                if delta.content:
                    yield ResponseResponse(
                        response_id=request.response_id,
                        content=delta.content,
                        content_complete=False,
                        end_call=False,
                    )

                # 2) Collect function name and arguments
                if getattr(delta, "function_call", None):
                    if delta.function_call.name:
                        func_name = delta.function_call.name
                    if delta.function_call.arguments:
                        func_args += delta.function_call.arguments

                # 3) Once the model finishes calling the function, handle it
                if chunk.choices[0].finish_reason == "function_call":
                    try:
                        fargs = json.loads(func_args)
                    except json.JSONDecodeError:
                        logger.exception("Failed to parse function_call JSON: %s", func_args)
                        yield ResponseResponse(
                            response_id=request.response_id,
                            content="Sorry, I couldn’t understand the request.",
                            content_complete=True,
                            end_call=False,
                        )
                        return

                    logger.info("🛠 Executing %s with %s", func_name, fargs)

                    if func_name == "knowledge_base_search":
                        result = await asyncio.to_thread(
                            self.rag_handler.search_documents_with_links, fargs.get("query", "")
                        )
                        async for evt in self.handle_tool_call_response(
                            request,
                            function_name=func_name,
                            original_query=fargs.get("query", ""),
                            tool_result=result
                        ):
                            yield evt

                    elif func_name == "end_call":
                        msg = fargs.get("message", "Thank you, goodbye!")
                        yield ResponseResponse(
                            response_id=request.response_id,
                            content=msg,
                            content_complete=True,
                            end_call=True,
                        )
                    else:
                        yield ResponseResponse(
                            response_id=request.response_id,
                            content=f"Error: Unknown function '{func_name}'",
                            content_complete=True,
                            end_call=False,
                        )
                    return

            # no function call: end turn
            yield ResponseResponse(
                response_id=request.response_id,
                content="",
                content_complete=True,
                end_call=False,
            )

        except Exception:
            logger.exception("Error during draft_response")
            yield ResponseResponse(
                response_id=request.response_id,
                content="Error! Check server logs for details.",
                content_complete=True,
                end_call=False,
            )
