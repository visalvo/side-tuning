import os
from pathlib import Path
from PIL import Image, ImageFont, ImageDraw

LINES_PER_IMAGE = 30
IMAGE_SIZE = 384
FONT_SIZE = 12
B_COLOR = (0, 0, 0)
F_COLOR = (255, 255, 255)


def render_text(text: str, filename: Path, font: ImageFont):
    img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), B_COLOR)

    d = ImageDraw.Draw(img)
    d.text((2, 2), text, font=font, fill=F_COLOR, spacing=1)

    img.save(filename)


def render_par(i):
    with open(INPUT_DIR_PATH + lang + '/' + i, 'r', encoding='utf-8') as file:
        content = file.readlines()
    text = ''.join(content[0:80])
    i = i[:-4]
    render_text(text, OUTPUT_DIR_PATH + lang + '/' + i + '.jpg', font)


font = ImageFont.truetype('/Users/salvo/Documents/tesi/datasets/SourceCodePro-Regular.ttf', FONT_SIZE)

OUTPUT_DIR_PATH = '/Users/salvo/Documents/tesi/datasets/imgs-47/'
INPUT_DIR_PATH = '/Users/salvo/Documents/tesi/datasets/snippets_2000_47_score-2_acc-false_rows-2/'

languages = []
for lang in os.listdir(INPUT_DIR_PATH):
    languages.append(lang)

if not os.path.exists(OUTPUT_DIR_PATH):
    os.mkdir(OUTPUT_DIR_PATH)

z = 1
for lang in languages:
    print(f'{str(z)}. {lang}')

    if not os.path.exists(os.path.join(OUTPUT_DIR_PATH, lang)):
        os.mkdir(os.path.join(OUTPUT_DIR_PATH, lang))

    X = os.listdir(os.path.join(INPUT_DIR_PATH, lang))

    for i in X:
        render_par(i)

    z += 1