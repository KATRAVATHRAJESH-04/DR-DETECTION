import base64
import os

img_path = r'c:\Users\ASUS\OneDrive\Desktop\project\bg.png'
ui_path = r'c:\Users\ASUS\OneDrive\Desktop\project\app\ui.py'

if os.path.exists(img_path):
    with open(img_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    
    with open(ui_path, 'r', encoding='utf-8') as f:
        content = f.read()

    style = f'''.stApp {{
        background-image: url("data:image/png;base64,{b64}");
        background-size: cover !important;
        background-attachment: fixed !important;
        background-position: center !important;
    }}
    .main {{ background: transparent !important; }}
    '''
    
    # Check if we already applied a .stApp style with b64
    if '.stApp {' in content and 'data:image/png;base64,' in content:
        # We need to replace the old block
        import re
        content = re.sub(r'\.stApp \{.*?\}\r?\n\s+\.main \{.*?\}', style, content, flags=re.DOTALL)
    else:
        target = ".main, .stApp { background-color: var(--bg-primary); }"
        content = content.replace(target, style)
    
    with open(ui_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Background updated successfully.")
else:
    print("Image not found at:", img_path)
