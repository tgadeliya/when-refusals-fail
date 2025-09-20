from pathlib import Path
import random

repo_root = Path(__file__).resolve().parent.parent.parent.parent

assets_dir = repo_root / "src/mountain/agentharm/paddings/non_relevant_text"

file_map = {
    "shortest_58": "shortest.txt",
    "labyrinth_433": "labyrinth_of_whispers.txt",
    "comedian_20397": "comedians_log.txt",
    "pineapple_essay1283": "pineapple_essay.txt",
    "unpuzzle_1932": "unpuzzle.txt",
    "unstory_701": "unstory.txt",
    "uneconomic_1275": "uneconomic_reforms.txt",
    "culinary_1900": "ministry_of_culinary.txt",
    "the_girl_from_scotland_yard_65017": "the_girl_from_scotland_yard.txt",
    "the_lady_with_the_dog_74379": "the_lady_with_the_dog.txt",
    "ill_kill_you_tomorrow_5600": "ill_kill_you_tomorrow.txt",
    "the_gift_of_magi_4973": "the_gift_of_magi.txt",
}

texts = {}
for var_name, filename in file_map.items():
    with open(assets_dir / filename, "r") as f:
        texts[var_name] = f.read()

padding_texts = {
    "100": texts["shortest_58"],
    "500": texts["labyrinth_433"],
    "1000": f"{texts['unstory_701']}\n{texts['labyrinth_433']}",
    "2000": f"{texts['pineapple_essay1283']}\n{texts['unstory_701']}",
    "5000": f"{texts['shortest_58']}\n{texts['the_gift_of_magi_4973']}",
    "10000": f"{texts['pineapple_essay1283']}\n{texts['unpuzzle_1932']}\n"
             f"{texts['culinary_1900']}\n{texts['the_gift_of_magi_4973']}",
    "20000": texts["comedian_20397"],
    "30000": f"{texts['comedian_20397']}\n{texts['ill_kill_you_tomorrow_5600']}\n"
             f"{texts['the_gift_of_magi_4973']}",
    "50000": f"{texts['comedian_20397']}\n{texts['ill_kill_you_tomorrow_5600']}\n"
             f"{texts['comedian_20397']}\n{texts['the_gift_of_magi_4973']}",
    "100000": f"{texts['comedian_20397']}\n{texts['the_lady_with_the_dog_74379']}\n"
              f"{texts['culinary_1900']}\n{texts['unpuzzle_1932']}\n"
              f"{texts['unstory_701']}\n{texts['pineapple_essay1283']}",
    "200000": f"{texts['comedian_20397']}\n{texts['the_girl_from_scotland_yard_65017']}\n"
              f"{texts['the_lady_with_the_dog_74379']}\n{texts['comedian_20397']}\n"
              f"{texts['ill_kill_you_tomorrow_5600']}\n{texts['pineapple_essay1283']}\n"
              f"{texts['the_gift_of_magi_4973']}\n{texts['unpuzzle_1932']}\n"
              f"{texts['culinary_1900']}\n{texts['the_gift_of_magi_4973']}",
}


# -- New set of fiction texts, divided by genre --
# -- Each text contains over 1M English words --
file_map = {
    "scifi": "scifi_ext.txt",
    "crime": "crime_ext.txt",
    "humour": "humour_ext.txt",
    "myth": "myth_ext.txt",
    "romance": "romance_ext.txt",
}
texts = {}
for genre, filename in file_map.items():
    with open(assets_dir / filename, "r") as f:
        texts[genre] = f.read()

def select_padding_text(padding_length: int, random_seed: int = 0):
    if random_seed is not None:
        random.seed(random_seed)
    genre = random.choice(list(texts.keys()))
    text = texts[genre]
    padding_length_chars = padding_length * 4 # rule of thumb: 4 letters are roughly one token
    starting_index = random.randint(0, len(text)-padding_length_chars)
    end_index = starting_index + padding_length_chars
    padding_text = text[starting_index:end_index]
    # assert len(padding_text) == padding_length * 4
    return padding_text


    