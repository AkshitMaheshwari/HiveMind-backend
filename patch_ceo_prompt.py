"""Patch script: fix CEO routing prompt for GitHub tasks."""
import pathlib

ceo_file = pathlib.Path(__file__).parent / "orchestrator" / "ceo_agent.py"
content = ceo_file.read_text(encoding="utf-8")

# ── 1. Fix CODE DEPARTMENT description ────────────────────────────────────────
old_code = (
    "\U0001f4bb CODE DEPARTMENT\n"
    "- Code generation, debugging, documentation, technical problem solving.\n"
    "- Use when: user wants code written, debugged, or explained."
)
new_code = (
    "\U0001f4bb CODE DEPARTMENT\n"
    "- Code generation, debugging, documentation, technical problem solving.\n"
    "- GitHub operations: show repo tree/structure, list repos, read code files, create branches, commit, open PRs.\n"
    "- Use when: user wants code written/debugged/explained, OR asks about a GitHub repository (structure, files, branches, PRs)."
)

if old_code in content:
    content = content.replace(old_code, new_code)
    print("CODE DEPT patched OK")
else:
    print("ERROR: CODE DEPT old text not found!")
    # Show what's there
    idx = content.find("CODE DEPARTMENT")
    print(repr(content[idx:idx+300]))

# ── 2. Add GitHub few-shot examples ───────────────────────────────────────────
old_end = (
    '- "hello" \u2192 clarification_needed=True, sequence="sequential", departments=[]\n'
    '- "what is in my uploaded PDF" \u2192 departments=["document"], sequence="sequential"\n'
    '"""'
)
new_end = (
    '- "hello" \u2192 clarification_needed=True, sequence="sequential", departments=[]\n'
    '- "what is in my uploaded PDF" \u2192 departments=["document"], sequence="sequential"\n'
    '- "show me the structure of my repo mailGraph" \u2192 departments=["code"], sequence="parallel"\n'
    '- "show me the project structure of AkshitMaheshwari/portfolio" \u2192 departments=["code"], sequence="parallel"\n'
    '- "list my GitHub repositories" \u2192 departments=["code"], sequence="parallel"\n'
    '- "read the main.py in owner/repo" \u2192 departments=["code"], sequence="parallel"\n'
    '- "create a pull request in my repo" \u2192 departments=["code"], sequence="parallel"\n'
    '- "what files are in owner/repo?" \u2192 departments=["code"], sequence="parallel"\n'
    '- "inspect my repository" \u2192 departments=["code"], sequence="parallel"\n'
    '"""'
)

if old_end in content:
    content = content.replace(old_end, new_end)
    print("Examples patched OK")
else:
    print("ERROR: Examples old text not found!")
    idx = content.find('"hello"')
    print(repr(content[idx-10:idx+300]))

ceo_file.write_text(content, encoding="utf-8")
print("File saved OK, size:", len(content))

# ── 3. Verify ──────────────────────────────────────────────────────────────────
from orchestrator.ceo_agent import CEO_SYSTEM_PROMPT
checks = {
    "github ops in CODE DEPT": "GitHub operations" in CEO_SYSTEM_PROMPT,
    "mailGraph example": "mailGraph" in CEO_SYSTEM_PROMPT,
    "list repos example": "list my GitHub" in CEO_SYSTEM_PROMPT,
    "inspect my repository": "inspect my repository" in CEO_SYSTEM_PROMPT,
}
print("\nVerification:")
for key, val in checks.items():
    print(f"  [{'OK' if val else 'FAIL'}] {key}")
