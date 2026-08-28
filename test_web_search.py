from server.web_search import (
    GAME_PROFILES,
    _build_game_search_queries,
    _game_result_relevant,
    _source_quality,
    detect_game_profile,
    is_live_query,
)


def demo():
    assert not _game_result_relevant(
        "https://wildrift.leagueoflegends.com/en-us/news/patch-notes/",
        "League of Legends patch notes",
        GAME_PROFILES["lol"],
    )
    assert _game_result_relevant(
        "https://www.leagueoflegends.com/en-us/news/game-updates/",
        "Patch notes",
        GAME_PROFILES["lol"],
    )
    assert not _game_result_relevant(
        "https://www.hoyolab.com/article/123",
        "Wuthering Waves update",
        GAME_PROFILES["genshin"],
    )
    assert _game_result_relevant(
        "https://www.hoyolab.com/article/123",
        "Genshin Impact update",
        GAME_PROFILES["genshin"],
    )
    assert _game_result_relevant(
        "https://www.pocketpair.jp/en/games-en/palworld-en/",
        "Palworld",
        GAME_PROFILES["palworld"],
    )
    apotheosis = GAME_PROFILES["minecraft_apotheosis"]
    prompt = "apotheosis 모드 몬스터팜 만드는법"
    assert detect_game_profile(prompt) == "minecraft_apotheosis"
    assert is_live_query(prompt)
    assert _build_game_search_queries(prompt, "minecraft_apotheosis") == [
        ("Apotheosis Minecraft mod guide spawner", "trusted"),
    ]
    assert _game_result_relevant(
        "https://github.com/Shadows-of-Fire/Apothic-Spawners",
        "Apothic Spawners",
        apotheosis,
    )
    assert _game_result_relevant(
        "https://www.curseforge.com/minecraft/mc-mods/apotheosis",
        "Apotheosis",
        apotheosis,
    )
    assert _source_quality(
        "https://github.com/Shadows-of-Fire/Apothic-Spawners",
        apotheosis,
    ) == "official"
    assert _source_quality(
        "https://minecraft-apotheosis-mod.fandom.com/wiki/Spawners",
        apotheosis,
    ) == "wiki"
    assert not _game_result_relevant(
        "https://github.com/other-project/apotheosis-tools",
        "Apotheosis tools",
        apotheosis,
    )
    assert not _game_result_relevant(
        "https://www.curseforge.com/minecraft/mc-mods/other-mod",
        "Apotheosis compatibility",
        apotheosis,
    )
    assert not _game_result_relevant(
        "https://www.curseforge.com/minecraft/mc-mods/apotheosis-fake",
        "Apotheosis fake project",
        apotheosis,
    )


if __name__ == "__main__":
    demo()
    print("GAME_SEARCH_FILTER_OK")
