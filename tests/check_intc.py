import requests, re
from html.parser import HTMLParser

HEADERS = {"User-Agent": "Supply Chain Risk Research hetp2030@gmail.com"}

r = requests.get(
    "https://www.sec.gov/Archives/edgar/data/50863/000005086323000006/intc-20221231.htm",
    headers=HEADERS, timeout=90
)
html = r.text
html = re.sub(r"<ix:header\b[^>]*>.*?</ix:header>", "", html, flags=re.DOTALL|re.IGNORECASE)
html = re.sub(r'<div[^>]+display\s*:\s*none[^>]*>.*?</div>', "", html, flags=re.DOTALL|re.IGNORECASE)

class S(HTMLParser):
    def __init__(self): super().__init__(); self.parts=[]
    def handle_data(self,d): self.parts.append(d)

p = S(); p.feed(html)
text = re.sub(r"\s+", " ", " ".join(p.parts)).strip()

print("=== first 600 chars ===")
print(repr(text[:600]))
print()
print("=== chars 35000-35500 ===")
print(repr(text[35000:35500]))
print()
# Find "Part I" or "Business" section markers
for pat in [r"PART\s*I\b", r"Part\s+I\b", r"BUSINESS\b"]:
    for m in list(re.finditer(pat, text, re.IGNORECASE))[:3]:
        print(f"pat={pat!r} pos={m.start():6d}: {repr(text[m.start():m.start()+150])}")
    print()
