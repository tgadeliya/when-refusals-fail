from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent.parent

assets_dir = repo_root / "assets"

with open(assets_dir / 'copyright.txt', 'r') as f:
    copyright = f.read()

with open(assets_dir / 'cybercrime.txt', 'r') as f:
    cybercrime = f.read()

with open(assets_dir / 'disinformation.txt', 'r') as f:
    disinformation = f.read()

with open(assets_dir / 'drugs.txt', 'r') as f:
    drugs = f.read()

with open(assets_dir / 'fraud.txt', 'r') as f:
    fraud = f.read()

with open(assets_dir / 'harassment.txt', 'r') as f:
    harassment = f.read()

with open(assets_dir / 'hate.txt', 'r') as f:
    hate = f.read()

with open(assets_dir / 'sexual.txt', 'r') as f:
    sexual = f.read()


CATEGORIES = {
    "Copyright": copyright,
    "Cybercrime": cybercrime,
    "Disinformation": disinformation,
    "Drugs": drugs,
    "Fraud": fraud,
    "Harassment": harassment,
    "Hate": hate,
    "Sexual": sexual,
}

CHARS_PER_TOKEN = 4

def build_padding(text: str, tokens: int) -> str:
    target_chars = tokens * CHARS_PER_TOKEN
    if len(text) >= target_chars:
        return text[:target_chars]

    reps = (target_chars // len(text)) + 1
    return (text * reps)[:target_chars]