import numpy as np
from PIL import Image, ImageDraw

# Create a mock Retinal Fundus image (dark background, red/orange circle)
img = Image.new('RGB', (800, 800), color=(10, 10, 10))
draw = ImageDraw.Draw(img)

# Draw the main retina circle
draw.ellipse((50, 50, 750, 750), fill=(180, 70, 40))

# Draw some "blood vessels" and a "macula" just for flavor
draw.ellipse((200, 350, 250, 400), fill=(120, 30, 20)) # Macula
draw.line((400, 400, 600, 200), fill=(100, 20, 10), width=10)
draw.line((400, 400, 650, 500), fill=(100, 20, 10), width=8)

# Add some "exudates" (yellowish spots that the model might detect)
draw.ellipse((500, 300, 520, 320), fill=(240, 220, 100))
draw.ellipse((550, 350, 560, 360), fill=(240, 220, 100))

img.save('test_eye.png')
print("Successfully generated test_eye.png!")
