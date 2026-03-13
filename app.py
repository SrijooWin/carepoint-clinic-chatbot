import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from groq import Groq

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ── CONFIG ────────────────────────────────────────────
GROQ_API_KEY = ""
client = Groq(api_key=GROQ_API_KEY)
MODEL  = "llama-3.1-8b-instant"

# ── CLINIC KNOWLEDGE BASE ─────────────────────────────
CLINIC_INFO = """
CLINIC NAME: CarePoint Clinic
DEPARTMENTS: General, Dental, Dermatology
DOCTORS:
  - Dr. Ramesh  → General    (fever, cold, headache, body pain)
  - Dr. Priya   → Dental     (tooth pain, gum problem, toothache)
  - Dr. Karthik → Dermatology (skin rash, acne, hair fall)
WORKING HOURS: Monday to Saturday, 9:00 AM to 6:00 PM
AVAILABLE SLOTS: 9am, 10am, 11am, 2pm, 3pm, 4pm, 5pm
CONSULTATION FEES:
  - General      → Rs.300
  - Dental       → Rs.400
  - Dermatology  → Rs.500
LOCATION: Anna Nagar, Chennai
"""

# ── SYSTEM PROMPT ─────────────────────────────────────
SYSTEM_PROMPT = f"""
You are a smart medical clinic chatbot assistant for CarePoint Clinic.
You help patients with appointment booking, doctor information, timings, and fees.

Here is the clinic information you must use:
{CLINIC_INFO}

Your behavior rules:
1. Always be polite, friendly, and caring
2. Keep responses short and clear
3. Ask only ONE question at a time
4. When patient mentions a symptom, suggest the right department and doctor
5. When booking, collect department, day, and time slot step by step
6. Confirm appointment details before finalizing
7. Only answer clinic related questions
8. If asked anything unrelated to clinic, politely redirect

Appointment booking flow:
Step 1 → Ask which department (or detect from symptom)
Step 2 → Ask which day
Step 3 → Ask which time slot
Step 4 → Confirm all details and finalize
"""

# ── NLP STEP 1: TOKENIZATION ──────────────────────────
def tokenize_text(text):
    tokens = word_tokenize(text.lower())
    return tokens

# ── NLP STEP 2: STOP WORD REMOVAL ────────────────────
def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    filtered = [w for w in tokens if w not in stop_words]
    return filtered

# ── NLP STEP 3: ENTITY EXTRACTION ────────────────────
def extract_entities(text, tokens):
    entities = {}

    # Extract time using regex
    time_pattern = re.findall(r'\b\d{1,2}(?:am|pm)\b', text.lower())
    if time_pattern:
        entities['time'] = time_pattern[0]

    # Extract day
    days = ['today', 'tomorrow', 'monday', 'tuesday', 'wednesday',
            'thursday', 'friday', 'saturday']
    for day in days:
        if day in tokens:
            entities['day'] = day

    # Extract department
    if 'general' in tokens:
        entities['department'] = 'General'
    elif any(w in tokens for w in ['dental', 'tooth', 'toothache', 'gum']):
        entities['department'] = 'Dental'
    elif any(w in tokens for w in ['dermatology', 'skin', 'rash', 'acne', 'hair']):
        entities['department'] = 'Dermatology'

    # Extract symptoms
    if 'tooth' in tokens:
        entities['symptom'] = 'tooth pain'
        entities['department'] = 'Dental'
    else:
        symptoms = {
            'fever'    : 'General',
            'headache' : 'General',
            'cold'     : 'General',
            'cough'    : 'General',
            'pain'     : 'General',
            'toothache': 'Dental',
            'gum'      : 'Dental',
            'rash'     : 'Dermatology',
            'acne'     : 'Dermatology',
            'hair'     : 'Dermatology',
            'skin'     : 'Dermatology'
        }
        for symptom, dept in symptoms.items():
            if symptom in tokens:
                entities['symptom'] = symptom
                if 'department' not in entities:
                    entities['department'] = dept
                break

    return entities

# ── NLP PIPELINE ──────────────────────────────────────
def nlp_pipeline(user_input):
    tokens   = tokenize_text(user_input)
    filtered = remove_stopwords(tokens)
    entities = extract_entities(user_input, filtered)
    return {
        'original': user_input,
        'tokens'  : tokens,
        'filtered': filtered,
        'entities': entities
    }

# ── LLM RESPONSE GENERATION ───────────────────────────
def get_llm_response(user_input, nlp_result, chat_history):

    # Build structured input combining user message + NLP analysis
    structured_input = f"""
Patient message: {nlp_result['original']}
NLP Analysis:
  - Keywords extracted: {nlp_result['filtered']}
  - Entities found: {nlp_result['entities']}

Please respond as the CarePoint Clinic assistant.
"""

    # Build full message history for context
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add last 6 messages for context memory
    for msg in chat_history[-6:]:
        messages.append({
            "role"   : msg['role'],
            "content": msg['content']
        })

    # Add current structured input
    messages.append({
        "role"   : "user",
        "content": structured_input
    })

    try:
        response = client.chat.completions.create(
            model      = MODEL,
            messages   = messages,
            max_tokens = 200,
            temperature= 0.7
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Sorry, something went wrong: {str(e)}"

# ── STREAMLIT UI ──────────────────────────────────────
st.set_page_config(page_title='CarePoint Clinic', page_icon='🏥')
st.title('🏥 CarePoint Clinic Chatbot')
st.caption('Powered by NLP preprocessing + Groq LLaMA 3 LLM')

show_nlp = st.toggle('🔬 Show NLP Analysis', value=True)

if 'messages' not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        'role'   : 'assistant',
        'content': 'Hi! Welcome to CarePoint Clinic 👋 How can I help you today?'
    })

for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

if user_input := st.chat_input('Type your message...'):

    # Run NLP pipeline
    nlp_result = nlp_pipeline(user_input)

    # Show NLP analysis
    if show_nlp:
        with st.expander('🔬 NLP Analysis', expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('**Tokens:**')
                st.write(nlp_result['tokens'])
                st.markdown('**After Stop Word Removal:**')
                st.write(nlp_result['filtered'])
            with col2:
                st.markdown('**Entities Extracted:**')
                st.json(nlp_result['entities'])
                st.markdown('**LLM Model:**')
                st.info('LLaMA 3 via Groq')

    # Show user message
    st.session_state.messages.append({
        'role'   : 'user',
        'content': user_input
    })
    with st.chat_message('user'):
        st.markdown(user_input)

    # Get LLM response
    with st.chat_message('assistant'):
        with st.spinner('Thinking...'):
            bot_reply = get_llm_response(
                user_input,
                nlp_result,
                st.session_state.messages
            )
            st.markdown(bot_reply)

    st.session_state.messages.append({
        'role'   : 'assistant',
        'content': bot_reply
    })
