from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent.parent

assets_dir = repo_root / "assets"

with open(assets_dir / "shortest.txt", "r") as f:
    shortest_58 = f.read()

with open(assets_dir / "labyrinth_of_whispers.txt", "r") as f:
    labyrinth_433 = f.read()

with open(assets_dir / "comedians_log.txt", "r") as f:
    comedian_20397 = f.read()

with open(assets_dir / "pineapple_essay.txt", "r") as f:
    pineapple_essay1283 = f.read()

with open(assets_dir / "unpuzzle.txt", "r") as f:
    unpuzzle_1932 = f.read()

with open(assets_dir / "unstory.txt", "r") as f:
    unstory_701 = f.read()

with open(assets_dir / "uneconomic_reforms.txt", "r") as f:
    uneconomic_1275 = f.read()

with open(assets_dir / "ministry_of_culinary.txt", "r") as f:
    culinary_1900 = f.read()

with open(assets_dir / "the_girl_from_scotland_yard.txt", "r") as f:
    the_girl_from_scotland_yard_65017 = f.read()

with open(assets_dir / "the_lady_with_the_dog.txt", "r") as f:
    the_lady_with_the_dog_74379 = f.read()

with open(assets_dir / "ill_kill_you_tomorrow.txt", "r") as f:
    ill_kill_you_tomorrow_5600 = f.read()

with open(assets_dir / "the_gift_of_magi.txt", "r") as f:
    the_gift_of_magi_4973 = f.read()


padding_texts = {
    "100": shortest_58,
    "500": f"{labyrinth_433}",
    "1000": f"{unstory_701}\n{labyrinth_433}",
    "2000": f"{pineapple_essay1283}\n{unstory_701}",
    "5000": f"{shortest_58}\n{the_gift_of_magi_4973}",
    "10_000": f"{pineapple_essay1283}\n{unpuzzle_1932}\n{culinary_1900}\n{the_gift_of_magi_4973}",
    "20_000": f"{comedian_20397}",
    "30_000": f"{comedian_20397}\n{ill_kill_you_tomorrow_5600}\n{the_gift_of_magi_4973}",
    "50_000": f"{comedian_20397}\n{ill_kill_you_tomorrow_5600}\n{comedian_20397}\n{the_gift_of_magi_4973}",
    "100_000": f"{comedian_20397}\n{the_lady_with_the_dog_74379}\n{culinary_1900}\n{unpuzzle_1932}\n{unstory_701}\n{pineapple_essay1283}",
    "200_000": f"{comedian_20397}\n{the_girl_from_scotland_yard_65017}\n{the_lady_with_the_dog_74379}\n{comedian_20397}\n{ill_kill_you_tomorrow_5600}\n{pineapple_essay1283}\n{the_gift_of_magi_4973}\n{unpuzzle_1932}\n{culinary_1900}\n{the_gift_of_magi_4973}",
}
