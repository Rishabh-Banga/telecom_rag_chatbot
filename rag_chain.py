"""
Builds the RAG chain:

merged retriever
    ↓
prompt
    ↓
Primary LLM
    ↓
Fallback LLM
    ↓
string output

The LLM provider/model is configurable and supports automatic fallback.
"""

import os

from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_core.documents import Document

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

from retriever import build_retriever


load_dotenv()


SYSTEM_PROMPT = """You are a helpful and professional telecom customer care assistant.

Your job is to help customers resolve technical issues with their mobile service.

Use ONLY the context below to answer the customer's question.

The context comes from two sources:
- FAQ entries (general policy and how-to information)
- Past support tickets (real resolved cases with step-by-step resolutions)

If the context does not contain enough information to answer confidently, say so clearly
and suggest the customer call 611 or use the MyTelecom app.

Context:
{context}
"""


def _format_docs(docs: list[Document]) -> str:
    sections = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown").upper()
        sections.append(
            f"[{source}]\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(sections)


def build_llms():
    """
    Build the available LLMs.

    The primary model is selected using PRIMARY_MODEL.

    Supported values:
        PRIMARY_MODEL=groq
        Secondary_MODEL=gemini

    The other provider automatically becomes the fallback.
    """

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")

    primary_model = os.getenv(
        "PRIMARY_MODEL",
        "groq"
    ).lower()

    llms = {}

    # ---------------------------------------------------------
    # Gemini
    # ---------------------------------------------------------

    if gemini_api_key:
        llms["gemini"] = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=gemini_api_key,
            temperature=0,
            max_retries=1,
        )

    # ---------------------------------------------------------
    # Groq
    # ---------------------------------------------------------

    if groq_api_key:
        llms["groq"] = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=1,
        )

    if not llms:
        raise RuntimeError(
            "No LLM API keys configured. "
            "Set GEMINI_API_KEY and/or GROQ_API_KEY."
        )

    # ---------------------------------------------------------
    # Determine primary + fallback
    # ---------------------------------------------------------

    if primary_model not in llms:
        available = ", ".join(llms.keys())

        raise ValueError(
            f"PRIMARY_MODEL='{primary_model}' is not available. "
            f"Available providers: {available}"
        )

    primary_llm = llms[primary_model]

    fallback_llms = [
        llm
        for provider, llm in llms.items()
        if provider != primary_model
    ]

    # ---------------------------------------------------------
    # Add fallback behavior
    # ---------------------------------------------------------

    if fallback_llms:
        primary_llm = primary_llm.with_fallbacks(
            fallback_llms
        )

    return primary_llm


def build_chain():

    # ---------------------------------------------------------
    # Retriever
    # ---------------------------------------------------------

    retriever = build_retriever()

    # ---------------------------------------------------------
    # Prompt
    # ---------------------------------------------------------

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    llm = build_llms()

    # ---------------------------------------------------------
    # RAG Chain
    # ---------------------------------------------------------

    chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain