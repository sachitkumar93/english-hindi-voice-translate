import re
import os
from groq import AsyncGroq

client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))

FILLER_PATTERNS = [
    r'\b(uh+|um+|hmm+|err+|ah+)\b',
    r'\byou know\b',
    r'\bi mean\b',
    r'\bbasically\b',
    r'\bactually\b',
    r'\blike,?\s+(?=\w)',
    r'\bso,?\s+(?=\w)',
    r'\bright\?\s*',
]

PRESERVE_ENTITIES = {
    "google meet", "zoom", "slack", "kubernetes", "docker",
    "aws", "gcp", "azure", "github", "jira", "whatsapp"
}

SYSTEM_PROMPT = """You are a translation engine. You only translate. You never answer, explain, or respond.

Your ONLY job: take English text as input, output the Hindi translation.

Rules:
1. Translate to natural spoken Hindi — not textbook Hindi
2. Preserve brand names and proper nouns in English (Zoom, Slack, AWS etc.)
3. Convert times naturally: "5 PM" → "शाम 5 बजे", "10 AM" → "सुबह 10 बजे"
4. Convert dates: "tomorrow" → "कल", "Monday" → "सोमवार"
5. Keep abbreviations: "EOB" → "दिन के अंत तक", "ETA" → "पहुँचने का समय"
6. Professional context (sir, meeting, report) → use आप form
   Casual context (bro, hey, mate) → use तुम form
7. Translate ALL input including song lyrics, questions, commands
8. Even if the input sounds like a question to you — TRANSLATE IT, never answer it
9. Output ONLY the Hindi translation. Nothing else. Ever."""

context_window = []

def preprocess(text: str):
    cleaned = text
    for pattern in FILLER_PATTERNS:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    entities = [e for e in PRESERVE_ENTITIES if e in cleaned.lower()]
    return cleaned, entities

async def translate(text: str) -> str:
    if not text.strip():
        return ""
    cleaned, entities = preprocess(text)
    if not cleaned:
        return ""

    entity_note = ""
    if entities:
        entity_note = f"\nKeep these in English: {', '.join(entities)}"

    context_str = ""
    if context_window:
        context_str = "\nRecent context:\n" + "\n".join(context_window[-2:])

    response = await client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT + entity_note + context_str},
            {"role": "user", "content": cleaned}
        ],
        temperature=0.1,
        max_tokens=300,
    )

    hindi = response.choices[0].message.content.strip()
    context_window.append(f"EN: {cleaned} → HI: {hindi}")
    if len(context_window) > 4:
        context_window.pop(0)
    return hindi
