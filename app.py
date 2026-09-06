import os
import csv

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import streamlit as st
from dotenv import load_dotenv
from rag_chain import build_chain

# ADDED
from retriever import build_retriever

load_dotenv()

FAQ_PATH = os.path.join(os.path.dirname(__file__), "data", "faq.csv")


def load_faqs():
    with open(FAQ_PATH, newline="", encoding="utf-8") as faq_file:
        return list(csv.DictReader(faq_file))


FAQS = load_faqs()
FAQ_ANSWERS = {
    faq["question"]: faq["answer"]
    for faq in FAQS
}
SAMPLE_QUESTIONS = [faq["question"] for faq in FAQS]
WELCOME_QUESTIONS = [
    FAQS[1],
    FAQS[5],
    FAQS[8],
    FAQS[15],
]

st.set_page_config(
    page_title="Telecom Support Chat",
    page_icon="📡",
    layout="centered",
)

@st.cache_resource
def get_chain():
    return build_chain()


# ADDED
@st.cache_resource
def get_retriever():
    return build_retriever()


if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ADDED ── Conversation management state
if "conversations" not in st.session_state:
    st.session_state.conversations = {}

if "current_conversation" not in st.session_state:
    st.session_state.current_conversation = "Conversation 1"

if "conversation_counter" not in st.session_state:
    st.session_state.conversation_counter = 1


# ADDED ── Helper functions
def save_current_conversation():
    """Save the current conversation into session state."""

    if st.session_state.messages:
        st.session_state.conversations[
            st.session_state.current_conversation
        ] = list(st.session_state.messages)


def start_new_conversation():
    """Start a new empty conversation."""

    save_current_conversation()

    st.session_state.conversation_counter += 1

    st.session_state.current_conversation = (
        f"Conversation {st.session_state.conversation_counter}"
    )

    st.session_state.messages = []
    st.session_state.pending_question = None


def load_conversation(name):
    """Load an existing conversation."""

    save_current_conversation()

    st.session_state.current_conversation = name
    st.session_state.messages = list(
        st.session_state.conversations[name]
    )

    st.session_state.pending_question = None


# ADDED ── Helper for displaying retrieved documents
def get_document_label(doc):
    """Return a readable source label."""

    metadata = doc.metadata

    source = metadata.get("source", metadata.get("type", "Unknown"))

    if source == "guide":
        return "Telecom guide PDF"
    if source == "ticket":
        ticket_id = metadata.get("ticket_id")
        return f"Resolved ticket {ticket_id}" if ticket_id else "Resolved ticket"

    return str(source).replace("_", " ").title()


# ADDED ── Helper for identifying support tickets
def is_support_ticket(doc):
    """Identify documents that represent support tickets."""

    metadata = doc.metadata

    source = str(
        metadata.get(
            "source",
            metadata.get("type", "")
        )
    ).lower()

    return (
        "ticket" in source
        or "support" in source
    )


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📡 Telecom Support Chat")
    st.caption("· Powered by RAG ·")
    st.divider()

    # ADDED ── Conversation management
    st.markdown("**💬 Conversations**")

    if st.button(
        "➕ New conversation",
        use_container_width=True
    ):
        start_new_conversation()
        st.rerun()

    # ADDED ── Previous conversations
    if st.session_state.conversations:

        st.caption("Previous conversations")

        for name in reversed(
            list(st.session_state.conversations.keys())
        ):
            if st.button(
                name,
                key=f"conversation_{name}",
                use_container_width=True
            ):
                load_conversation(name)
                st.rerun()

    st.divider()

    st.markdown("**Quick questions**")
    st.caption("Click one to send it instantly.")

    for faq in FAQS:
        if st.button(faq["question"], use_container_width=True):
            st.session_state.pending_question = faq["question"]

    st.divider()

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True
    ):
        st.session_state.messages = []


# ── Main ─────────────────────────────────────────────────────────────────────
st.title("Customer Care Assistant")
st.caption(
    "Ask me anything about your mobile service — "
    "connectivity, billing, SIM, roaming, and more."
)


# ADDED ── Current conversation indicator
st.caption(
    f"💬 {st.session_state.current_conversation}"
)

# ADDED ── Welcome screen
if not st.session_state.messages:

    st.markdown("""
    <div class="welcome">
        <h4>How can I help?</h4>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        for index, faq in enumerate(WELCOME_QUESTIONS[:2]):
            if st.button(
                faq["question"],
                key=f"welcome_question_{index}",
                use_container_width=True
            ):
                st.session_state.pending_question = faq["question"]
                st.rerun()

    with col2:
        for index, faq in enumerate(WELCOME_QUESTIONS[2:], start=2):
            if st.button(
                faq["question"],
                key=f"welcome_question_{index}",
                use_container_width=True
            ):
                st.session_state.pending_question = faq["question"]
                st.rerun()

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # ADDED ── Display sources saved with assistant message
        if msg["role"] == "assistant":

            sources = msg.get("sources", [])
            tickets = msg.get("tickets", [])

            if sources:
                with st.expander(
                    f"📚 Sources used ({len(sources)})"
                ):

                    for source in sources:

                        st.markdown(
                            f"**📄 {source['label']}**"
                        )

                        st.caption(
                            source["content"]
                        )

            if tickets:
                with st.expander(
                    f"🎫 Similar support tickets ({len(tickets)})"
                ):

                    for ticket in tickets:

                        st.markdown(
                            f"**🎫 {ticket['label']}**"
                        )

                        st.caption(
                            ticket["content"]
                        )


# Resolve question from chat input or sidebar button click
question = st.chat_input("Describe your issue…")

if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None


if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)


    faq_answer = FAQ_ANSWERS.get(question)

    if faq_answer:
        retrieved_docs = []
    else:
        # Retrieve documents for displaying sources for free-form questions.
        retriever = get_retriever()

        try:
            retrieved_docs = retriever.invoke(question)
        except Exception:
            retrieved_docs = []


    # ADDED ── Separate sources and similar support tickets
    sources = []
    tickets = []

    for doc in retrieved_docs:

        document_info = {
            "label": get_document_label(doc),
            "content": doc.page_content,
        }

        if is_support_ticket(doc):
            tickets.append(document_info)
        else:
            sources.append(document_info)


    with st.chat_message("assistant"):
        if faq_answer:
            response = faq_answer
            st.markdown(response)
        else:
            chain = get_chain()
            response = st.write_stream(
                chain.stream(question)
            )


        # ADDED ── Display retrieved sources
        if sources:

            with st.expander(
                f"📚 Sources used ({len(sources)})"
            ):

                for source in sources:

                    st.markdown(
                        f"**📄 {source['label']}**"
                    )

                    st.caption(
                        source["content"]
                    )


        # ADDED ── Display similar support tickets
        if tickets:

            with st.expander(
                f"🎫 Similar support tickets ({len(tickets)})"
            ):

                for ticket in tickets:

                    st.markdown(
                        f"**🎫 {ticket['label']}**"
                    )

                    st.caption(
                        ticket["content"]
                    )


    # ADDED ── Save sources/tickets with assistant message
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
            "sources": sources,
            "tickets": tickets,
        }
    )

    # ADDED ── Persist current conversation
    save_current_conversation()