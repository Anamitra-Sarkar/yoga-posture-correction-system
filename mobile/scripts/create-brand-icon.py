from PIL import Image, ImageDraw

SIZE = 1024
PAPER = "#F8F7F2"
INK = "#1F2621"
MOSS = "#5C6B4E"
TERRACOTTA = "#B86145"
SAGE = "#DDE2D8"

image = Image.new("RGBA", (SIZE, SIZE), PAPER)
draw = ImageDraw.Draw(image)

# A quiet, geometric yoga balance mark. The simplified posture remains legible at launcher sizes.
draw.ellipse((116, 116, 908, 908), fill=SAGE)
draw.ellipse((150, 150, 874, 874), outline=MOSS, width=30)

# Head and raised arm create a stable upright silhouette.
draw.ellipse((455, 244, 569, 358), fill=MOSS)
draw.rounded_rectangle((478, 365, 546, 612), radius=34, fill=MOSS)
draw.rounded_rectangle((342, 405, 516, 468), radius=32, fill=MOSS)
draw.rounded_rectangle((508, 405, 682, 468), radius=32, fill=MOSS)

# Grounded legs: one standing, one bent in a tree-like resting angle.
draw.rounded_rectangle((475, 582, 539, 782), radius=32, fill=MOSS)
draw.rounded_rectangle((525, 604, 680, 666), radius=31, fill=MOSS)
draw.ellipse((635, 592, 726, 683), fill=MOSS)
draw.rounded_rectangle((438, 750, 570, 814), radius=30, fill=MOSS)

# Warm accent: a subtle setting-sun curve, not a glow or gradient.
draw.arc((270, 360, 754, 844), start=30, end=115, fill=TERRACOTTA, width=38)

for output in [
    "assets/images/icon.png",
    "assets/images/splash-icon.png",
    "assets/images/favicon.png",
    "assets/images/android-icon-foreground.png",
]:
    image.save(output, "PNG")
