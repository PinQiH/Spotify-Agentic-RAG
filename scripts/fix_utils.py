
import os

path = r'd:\台北市立大學 資訊科學系 在職專班\碩一\上學期\資料探勘\final_spotify\utils.py'
try:
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
except UnicodeDecodeError:
    with open(path, 'r', encoding='cp950') as f: # Try default Windows encoding
        content = f.read()

target = '''            xanchor="center",
            x=0.5,
            title=None         # Remove title to prevent overlap
        ),'''

replacement = '''            xanchor="left",    # Left align to prevent centering overlap issues
            x=0,               # Start from left edge
            title=None         # Remove title to prevent overlap
        ),
        xaxis=dict(title="PC1", side="bottom"), # Force PC1 label to bottom
        yaxis=dict(title="PC2", side="left"),   # Force PC2 label to left'''

if target in content:
    new_content = content.replace(target, replacement)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("SUCCESS: File updated.")
else:
    print("TARGET NOT FOUND")
    # Debug: print surrounding area to see what's wrong
    marker = 'font=dict(color=\'white\'),'
    start_idx = content.find(marker)
    if start_idx != -1:
        print("Surrounding content:")
        print(repr(content[start_idx:start_idx+300]))
