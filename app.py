"""FastAPI app with a structured HTML form UI for the meal-planning agent."""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from agents.graph import build_graph
from langchain_core.messages import HumanMessage

app = FastAPI()
graph = build_graph()

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Meal Planner</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg: #f7f6f3;
    --panel: #ffffff;
    --border: #e5e3de;
    --accent: #2d6a4f;
    --accent-hover: #1b4332;
    --text: #1a1a1a;
    --muted: #6b7280;
    --radius: 10px;
  }

  body {
    font-family: system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    height: 100vh;
    display: flex;
    flex-direction: column;
  }

  header {
    padding: 16px 28px;
    border-bottom: 1px solid var(--border);
    background: var(--panel);
    display: flex;
    align-items: center;
    gap: 10px;
  }

  header h1 { font-size: 1.1rem; font-weight: 600; }
  header span { font-size: 1.4rem; }

  .layout {
    display: grid;
    grid-template-columns: 340px 1fr;
    flex: 1;
    overflow: hidden;
  }

  .sidebar {
    background: var(--panel);
    border-right: 1px solid var(--border);
    padding: 24px 20px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 18px;
  }

  .field { display: flex; flex-direction: column; gap: 6px; }

  label { font-size: 0.8rem; font-weight: 600; text-transform: uppercase;
          letter-spacing: 0.04em; color: var(--muted); }

  textarea, input[type="text"], input[type="number"] {
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 10px 12px;
    font-size: 0.95rem;
    font-family: inherit;
    background: var(--bg);
    color: var(--text);
    transition: border-color 0.15s;
    width: 100%;
  }
  textarea { resize: vertical; min-height: 100px; }
  textarea:focus, input:focus { outline: none; border-color: var(--accent); }

  .budget-wrap { position: relative; }
  .budget-wrap span {
    position: absolute; left: 12px; top: 50%; transform: translateY(-50%);
    color: var(--muted); font-size: 0.95rem; pointer-events: none;
  }
  .budget-wrap input { padding-left: 24px; }

  .checks { display: flex; flex-wrap: wrap; gap: 8px; }
  .check-pill {
    display: flex; align-items: center; gap: 6px;
    padding: 5px 12px; border-radius: 20px;
    border: 1px solid var(--border); background: var(--bg);
    font-size: 0.85rem; cursor: pointer; transition: all 0.15s;
    user-select: none;
  }
  .check-pill input { display: none; }
  .check-pill:has(input:checked) {
    background: var(--accent); color: #fff; border-color: var(--accent);
  }

  button[type="submit"] {
    margin-top: 4px;
    padding: 12px;
    background: var(--accent);
    color: #fff;
    border: none;
    border-radius: var(--radius);
    font-size: 0.95rem;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.15s;
  }
  button[type="submit"]:hover { background: var(--accent-hover); }
  button[type="submit"]:disabled { opacity: 0.6; cursor: default; }

  .results {
    padding: 32px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
  }

  .empty {
    flex: 1; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    color: var(--muted); gap: 10px; text-align: center;
  }
  .empty .icon { font-size: 3rem; }
  .empty p { font-size: 0.95rem; }

  .spinner {
    display: none; flex: 1; align-items: center; justify-content: center;
  }
  .spinner.active { display: flex; }
  .spinner-ring {
    width: 36px; height: 36px;
    border: 3px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  #output {
    display: none;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 28px 32px;
    line-height: 1.7;
    max-width: 720px;
  }
  #output.visible { display: block; }

  #output h1, #output h2, #output h3 {
    margin-top: 1.2em; margin-bottom: 0.4em; font-weight: 600;
  }
  #output h1 { font-size: 1.3rem; }
  #output h2 { font-size: 1.1rem; color: var(--accent); }
  #output h3 { font-size: 1rem; }
  #output p { margin-bottom: 0.75em; }
  #output ul, #output ol { padding-left: 1.4em; margin-bottom: 0.75em; }
  #output li { margin-bottom: 0.3em; }
  #output strong { font-weight: 600; }
  #output code {
    background: var(--bg); padding: 1px 5px;
    border-radius: 4px; font-size: 0.88em;
  }
  #output hr { border: none; border-top: 1px solid var(--border); margin: 1.2em 0; }

  .error-msg {
    color: #b91c1c; background: #fef2f2;
    border: 1px solid #fecaca; border-radius: var(--radius);
    padding: 14px 18px; font-size: 0.9rem;
    display: none; max-width: 500px;
  }
  .error-msg.visible { display: block; }
</style>
</head>
<body>

<header>
  <span>🥗</span>
  <h1>Meal Planner</h1>
</header>

<div class="layout">
  <aside class="sidebar">
    <form id="form">

      <div class="field">
        <label>What would you like?</label>
        <textarea name="message" placeholder="e.g. A quick weeknight dinner for two under $25" required></textarea>
      </div>

      <div class="field">
        <label>Allergies</label>
        <input type="text" name="allergies" placeholder="e.g. peanuts, shellfish">
      </div>

      <div class="field">
        <label>Budget</label>
        <div class="budget-wrap">
          <span>$</span>
          <input type="number" name="budget" placeholder="30" min="1">
        </div>
      </div>

      <div class="field">
        <label>Dietary preferences</label>
        <div class="checks">
          <label class="check-pill"><input type="checkbox" name="diet" value="vegetarian"> Vegetarian</label>
          <label class="check-pill"><input type="checkbox" name="diet" value="vegan"> Vegan</label>
          <label class="check-pill"><input type="checkbox" name="diet" value="gluten-free"> Gluten-free</label>
          <label class="check-pill"><input type="checkbox" name="diet" value="dairy-free"> Dairy-free</label>
        </div>
      </div>

      <button type="submit" id="btn">Generate plan</button>
    </form>
  </aside>

  <main class="results">
    <div class="empty" id="empty">
      <span class="icon">🍽️</span>
      <p>Fill in the form and click <strong>Generate plan</strong>.</p>
    </div>
    <div class="spinner" id="spinner"><div class="spinner-ring"></div></div>
    <div id="output"></div>
    <div class="error-msg" id="error"></div>
  </main>
</div>

<script>
  const form = document.getElementById('form');
  const btn = document.getElementById('btn');
  const empty = document.getElementById('empty');
  const spinner = document.getElementById('spinner');
  const output = document.getElementById('output');
  const errorEl = document.getElementById('error');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const message = form.message.value.trim();
    const allergies = form.allergies.value.trim();
    const budget = form.budget.value.trim();
    const diet = [...form.querySelectorAll('input[name="diet"]:checked')].map(el => el.value);

    empty.style.display = 'none';
    output.classList.remove('visible');
    errorEl.classList.remove('visible');
    spinner.classList.add('active');
    btn.disabled = true;

    try {
      const res = await fetch('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, allergies, budget, diet }),
      });

      if (!res.ok) throw new Error(`Server error ${res.status}`);
      const data = await res.json();

      output.innerHTML = marked.parse(data.response);
      output.classList.add('visible');
    } catch (err) {
      errorEl.textContent = `Something went wrong: ${err.message}`;
      errorEl.classList.add('visible');
    } finally {
      spinner.classList.remove('active');
      btn.disabled = false;
    }
  });
</script>

</body>
</html>
"""


class AskRequest(BaseModel):
    message: str
    allergies: str = ""
    budget: str = ""
    diet: list[str] = []


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


@app.post("/ask")
def ask(req: AskRequest):
    parts = [req.message]
    if req.allergies:
        parts.append(f"I'm allergic to: {req.allergies}")
    if req.budget:
        parts.append(f"Budget: ${req.budget}")
    if req.diet:
        parts.append(f"Dietary preferences: {', '.join(req.diet)}")

    full_message = ". ".join(parts)
    result = graph.invoke({"messages": [HumanMessage(content=full_message)]})
    return {"response": result["messages"][-1].content}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
