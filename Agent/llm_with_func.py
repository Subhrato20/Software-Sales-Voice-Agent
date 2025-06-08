import logging
import os
import json
import asyncio
from typing import List, Optional

from openai import AsyncOpenAI
from mem0 import MemoryClient
from .custom_types import ResponseRequiredRequest, ResponseResponse, Utterance
from RAG.milvus_search import RAGHandler

# — Configure module-wide logging —
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
)
logger = logging.getLogger(__name__)

# — Greeting templates —
BEGIN_SENTENCE_KNOWN = "Hi {name}, thanks for calling! How can I help you today?"
BEGIN_SENTENCE_UNKNOWN = "Hello! To get started, may I have your name?"

# — Core AI prompt —
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
        # — OpenAI client setup —
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set in the environment")
        self.client = AsyncOpenAI(api_key=api_key)
        self.model_name = os.getenv("GEMINI_MODEL", "gpt-4o")

        # — RAG handler for KB searches —
        self.rag_handler = RAGHandler()

        # — Mem0 client for long-term memory :contentReference[oaicite:0]{index=0}
        self.memory_client = MemoryClient(
            api_key=os.getenv("MEM0_API_KEY"),
            org_id=os.getenv("MEM0_ORG_ID"),
            project_id=os.getenv("MEM0_PROJECT_ID"),
        )
        self.user_id: Optional[str] = None

    def draft_begin_message(self) -> ResponseResponse:
        """
        On a brand-new call, ask for name if unknown or greet by name.
        """
        if not self.user_id:
            return ResponseResponse(
                response_id=0,
                content=BEGIN_SENTENCE_UNKNOWN,
                content_complete=True,
                end_call=False,
            )
        return ResponseResponse(
            response_id=0,
            content=BEGIN_SENTENCE_KNOWN.format(name=self.user_id),
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
        # — Capture name on first turn —
        if not self.user_id and request.transcript:
            candidate = request.transcript[-1].content.strip()
            self.user_id = candidate
            logger.info("Set user_id to '%s'", self.user_id)
            try:
                self.memory_client.add(
                    [{"role": "user", "content": f"My name is {self.user_id}"}],
                    user_id=self.user_id,
                )
            except Exception as e:
                logger.warning("Failed to store name in Mem0: %s", e)

        # — Retrieve up to 5 relevant memories :contentReference[oaicite:1]{index=1}
        memory_context: List[str] = []
        if self.user_id and request.transcript:
            last_user = request.transcript[-1].content.strip()
            try:
                mem_res = self.memory_client.search(
                    query=last_user,       # non-empty required :contentReference[oaicite:2]{index=2}
                    version="v2",
                    filters={"AND": [{"user_id": self.user_id}]},
                    limit=5
                )
                for m in mem_res:
                    # v2 returns objects with "memory" field
                    memory_context.append(m.get("memory") or m.get("data", {}).get("memory", ""))
            except Exception as e:
                logger.warning("Mem0 search failed, skipping memories: %s", e)

        # — Build system prompt with memories injected —
        system_content = AGENT_PROMPT
        if memory_context:
            snippets = "\n".join(f"- {mem}" for mem in memory_context)
            system_content = (
                "Here’s what I remember from our past chats:\n"
                f"{snippets}\n\n"
            ) + system_content

        system = {"role": "system", "content": system_content}
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
        Format raw KB output into a concise reply.
        """
        try:
            formatting_msgs = [
                {
                    "role": "system",
                    "content": (
                        "You are a professional, friendly AI assistant. "
                        f"The user asked: “{original_query}”.\n\n"
                        "Below is the information retrieved from the knowledge base. "
                        "Please condense it into a concise, accurate reply."
                    ),
                },
                {"role": "user", "content": tool_result},
            ]

            logger.debug("Formatting tool result…")
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

            # signal end of answer
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
        """
        Main response generator: handles openai function calls, RAG searches, 
        mem0 persistence, and streaming back to the caller.
        """
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
            collected_reply = ""

            async for chunk in stream:
                delta = chunk.choices[0].delta

                # — Plain text content —
                if delta.content:
                    collected_reply += delta.content
                    yield ResponseResponse(
                        response_id=request.response_id,
                        content=delta.content,
                        content_complete=False,
                        end_call=False,
                    )

                # — Function call metadata collection —
                if getattr(delta, "function_call", None):
                    if delta.function_call.name:
                        func_name = delta.function_call.name
                    if delta.function_call.arguments:
                        func_args += delta.function_call.arguments

                # — When function call completes → execute tool and format response —
                if chunk.choices[0].finish_reason == "function_call":
                    try:
                        fargs = json.loads(func_args)
                    except json.JSONDecodeError:
                        logger.exception("Invalid function_call JSON: %s", func_args)
                        yield ResponseResponse(
                            response_id=request.response_id,
                            content="Sorry, I couldn't understand that request.",
                            content_complete=True,
                            end_call=False,
                        )
                        return

                    logger.info("Executing %s with %s", func_name, fargs)

                    if func_name == "knowledge_base_search":
                        result = await asyncio.to_thread(
                            self.rag_handler.search_documents_with_links,
                            fargs.get("query", "")
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

            # — No function call → end of turn —
            yield ResponseResponse(
                response_id=request.response_id,
                content="",
                content_complete=True,
                end_call=False,
            )

            # — Persist turn into Mem0 :contentReference[oaicite:3]{index=3}
            if self.user_id and request.transcript:
                last_user = request.transcript[-1].content
                try:
                    self.memory_client.add(
                        [
                            {"role": "user", "content": last_user},
                            {"role": "assistant", "content": collected_reply}
                        ],
                        user_id=self.user_id,
                        version="v2",
                    )
                except Exception as e:
                    logger.warning("Failed to store turn in Mem0: %s", e)

        except Exception:
            logger.exception("Error during draft_response")
            yield ResponseResponse(
                response_id=request.response_id,
                content="Error! Check server logs for details.",
                content_complete=True,
                end_call=False,
            )
