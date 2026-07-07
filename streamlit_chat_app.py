import streamlit as st
import sqlite3
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

# ── 1. KONFIGURASI HALAMAN UTAMA ──────────────────────────────────────────────
st.set_page_config(page_title="Enterprise Multi-Agent AI", layout="wide", page_icon="🤖")
st.title("🤖 Multi-Agent AI Platform System")
st.caption("Aplikasi Multi-Agent cerdas terintegrasi menggunakan LangGraph ReAct Framework & Gemini 2.5.")

DB_FILE = "sample.db"

# ── 2. FUNGSI UTAMA AGEN & DATABASE ───────────────────────────────────────────

# --- Tools untuk SQL Agent ---
def list_tables() -> list[str]:
    """Retrieve the names of all tables in the database."""
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            return [t[0] for t in cursor.fetchall()]
    except Exception:
        return []

def describe_table(table_name: str) -> list[tuple[str, str]]:
    """Look up the table schema. Returns list of (column, type)."""
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name});")
            return [(col[1], col[2]) for col in cursor.fetchall()]
    except Exception:
        return []

def execute_query(sql: str) -> list[list[str]]:
    """Execute a SELECT statement, returning the results."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(sql)
        return cursor.fetchall()

# --- Tools & State untuk BaristaBot ---
if "customer_order" not in st.session_state:
    st.session_state.customer_order = []

def get_menu() -> str:
    """Provide the latest up-to-date menu."""
    return """
    MENU:
    - Coffee: Espresso, Americano, Cold Brew
    - Milk Coffee: Latte, Cappuccino, Cortado, Macchiato, Mocha, Flat White
    - Tea: English Breakfast Tea, Green Tea, Earl Grey
    - Milk Tea: Chai Latte, Matcha Latte, London Fog
    - Other: Steamer, Hot Chocolate
    - Modifiers: Milk (Whole, 2%, Oat, Almond), Espresso shots, Caffeine (Decaf, Regular), Hot-Iced.
    Catatan khusus: Susu Kedelai (Soy milk) HABIS/OUT OF STOCK hari ini.
    """

def add_to_order(item: str) -> str:
    """Add an item to the customer's order."""
    st.session_state.customer_order.append(item)
    return f"I've added '{item}' to your order."

def clear_order() -> str:
    """Clear all items from the customer's order."""
    st.session_state.customer_order.clear()
    return "Your order has been cleared."

def get_order() -> list[str]:
    """Get the current items in the customer's order."""
    return st.session_state.customer_order

def confirm_order() -> str:
    """Confirm the order with the customer."""
    if not st.session_state.customer_order:
        return "Your order is currently empty. What can I get for you?"
    return f"Your order contains: {', '.join(st.session_state.customer_order)}. Is this correct?"

def place_order() -> str:
    """Place the final order."""
    if not st.session_state.customer_order:
        return "There's nothing in your order to place."
    final_summary = ", ".join(st.session_state.customer_order)
    st.session_state.customer_order.clear()
    return f"Your order for '{final_summary}' has been placed! It will be ready shortly."


# ── 3. SIDEBAR DINAMIS & KONFIGURASI PENGGUNA ─────────────────────────────────
with st.sidebar:
    st.header("⚙️ Control Panel")
    google_api_key = st.text_input("🔑 Google AI API Key", type="password", value=os.environ.get("GOOGLE_API_KEY", ""))

    st.write("---")
    agent_mode = st.selectbox(
        "🧠 Pilih Mode Chatbot Agent:",
        ["General Gemini Chat", "SQL Store Agent", "BaristaBot Café"]
    )

    st.write("---")

    # —— KONTEN SIDEBAR DINAMIS BERDASARKAN AGEN YANG DIPILIH ——
    if agent_mode == "SQL Store Agent":
        st.subheader("🗄️ Database Explorer (Live)")
        tables = list_tables()
        if tables:
            for table in tables:
                with st.expander(f"📋 Tabel: {table}"):
                    columns = describe_table(table)
                    for col_name, col_type in columns:
                        st.write(f"`{col_name}` ({col_type})")
        else:
            st.caption("Database kosong atau file 'sample.db' belum dibuat.")

    elif agent_mode == "BaristaBot Café":
        st.subheader("🛒 Keranjang Belanja")
        if st.session_state.customer_order:
            for idx, item in enumerate(st.session_state.customer_order, 1):
                st.write(f"{idx}. ☕ **{item}**")
            if st.button("🗑️ Kosongkan Keranjang"):
                st.session_state.customer_order.clear()
                st.rerun()
        else:
            st.info("Keranjang Anda kosong. Silakan lakukan pemesanan di chat!")

    st.write("---")
    reset_button = st.button("🔄 Reset Total Percakapan", use_container_width=True)

if not google_api_key:
    st.warning("⚠️ Silakan masukkan Google AI API Key Anda pada Control Panel di sebelah kiri untuk mengaktifkan sistem AI.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = google_api_key

# ── 4. MANAJEMEN STATE & PENGALIHAN AGEN ──────────────────────────────────────
if "current_agent" not in st.session_state:
    st.session_state.current_agent = agent_mode
    st.session_state.messages = []

# Deteksi jika user mengganti jenis Agen di dropdown, lakukan reset otomatis agar tidak tabrakan konteks
if reset_button or (st.session_state.current_agent != agent_mode):
    st.session_state.messages = []
    st.session_state.customer_order = []
    st.session_state.pop("agent_executor", None)
    st.session_state.current_agent = agent_mode
    st.rerun()

# ── 5. INISIALISASI PEMBUATAN AGENT (LANGGRAPH REACT) ─────────────────────────
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

if agent_mode == "SQL Store Agent":
    instruction = "You are a helpful chatbot that can interact with an SQL database for a computer store. ALWAYS start by calling list_tables to discover the available tables. ALWAYS call describe_table on the relevant table before writing any SQL query. Never assume or guess table names or column names — always verify first."
    db_tools = [list_tables, describe_table, execute_query]
    st.session_state.agent_executor = create_react_agent(model=llm, tools=db_tools, prompt=instruction)

elif agent_mode == "BaristaBot Café":
    instruction = "You are BaristaBot, an interactive cafe ordering system. Use the tools to manage orders. Always confirm_order with the user before calling place_order. If you are unsure a drink or modifier matches those on the MENU, ask a question to clarify."
    barista_tools = [get_menu, add_to_order, clear_order, get_order, confirm_order, place_order]
    memory = InMemorySaver()
    st.session_state.agent_executor = create_react_agent(model=llm, tools=barista_tools, prompt=instruction, checkpointer=memory)

else:
    st.session_state.agent_executor = None

# ── 6. ANTARMUKA CHAT (UI RENDERING) ──────────────────────────────────────────
# Tampilkan seluruh chat log dari session state
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Ambil input baru dari pengguna
user_prompt = st.chat_input("Ketik pertanyaan Anda di sini...")

if user_prompt:
    # Tampilkan pesan user ke layar
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Proses respons oleh Agen pilihan
    with st.chat_message("assistant"):
        # Tambahkan animasi loading spinner agar interaksi lebih halus
        with st.spinner("Agen sedang menganalisis & menjalankan tools..."):
            try:
                if st.session_state.agent_executor:
                    # Jalankan via LangGraph ReAct Agent
                    config = {"configurable": {"thread_id": "streamlit_session_thread"}}
                    response = st.session_state.agent_executor.invoke(
                        {"messages": [{"role": "user", "content": user_prompt}]},
                        config=config
                    )
                    answer = response["messages"][-1].content
                else:
                    # Jalankan via Standar Gemini Chat
                    response = llm.invoke(user_prompt)
                    answer = response.content

                # Tampilkan hasil dan simpan ke riwayat
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

                # Rerun otomatis jika ada perubahan state internal (seperti keranjang barista bertambah)
                if agent_mode == "BaristaBot Café":
                    st.rerun()

            except Exception as e:
                st.error(f"Terjadi kesalahan pada sistem pemrosesan Agen: {e}")
