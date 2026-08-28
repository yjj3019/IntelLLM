from server.web_search import GAME_PROFILES, _game_result_relevant


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


if __name__ == "__main__":
    demo()
    print("GAME_SEARCH_FILTER_OK")
