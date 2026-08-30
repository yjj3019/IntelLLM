import html
import json
import os
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

os.environ["NO_PROXY"] = ",".join(dict.fromkeys(
    [part.strip() for part in os.getenv("NO_PROXY", "").split(",") if part.strip()]
    + ["192.168.1.3", "127.0.0.1", "localhost"]
))


SEARXNG_URL = os.environ.get(
    "LOCAL_SEARCH_URL",
    "http://192.168.1.3:8888",
).rstrip("/")

SEARCH_TIMEOUT = 8
WEATHER_TIMEOUT = 6

KST = timezone(
    timedelta(hours=9),
    name="KST",
)


LIVE_PATTERNS = (
    "날씨",
    "기온",
    "강수",
    "태풍",
    "기상",
    # "오늘"/"내일"/"이번 주" were removed: they matched on their own in
    # plain non-live prose ("오늘 배운 내용 정리해줘", a date-format
    # question containing "오늘 날짜 형식"), and every genuinely live
    # case they were meant to catch (오늘 뉴스, 오늘 환율, 오늘 날씨...)
    # is already covered by its own more specific keyword below.
    "최신",
    "최근",
    "실시간",
    "뉴스",
    "속보",
    "현재 시간",
    "현재 시각",
    "오늘 날짜",
    "날짜와 시간",
    "지금 몇 시",
    "주가",
    "환율",
    "웹 검색",
    "검색해",
    "검색해서",
    "인터넷",
    "공식 문서",
    "공식 링크",
    "출처",
    "링크",
    "web search",
    "search the web",
    "latest",
    "weather",
    "forecast",
    "temperature",
    "rain",
    "snow",
    "humidity",
    "typhoon",
    "storm",
    "current time",
    "what time is it",
    "what's the time",
)

WEATHER_PATTERNS = (
    "날씨",
    "기온",
    "강수",
    "태풍",
    "기상",
    "weather",
    "forecast",
    "temperature",
    "rain",
    "snow",
    "humidity",
    "typhoon",
    "storm",
)

TIME_PATTERNS = (
    "현재 시간",
    "현재 시각",
    "오늘 날짜",
    "날짜와 시간",
    "지금 몇 시",
    "몇 시",
    "몇시",
    "current time",
    "what time is it",
    "what's the time",
)

LOCATION_ALIASES = {
    "서울": (37.5665, 126.9780),
    "인천": (37.4563, 126.7052),
    "김포": (37.6153, 126.7156),
    "수원": (37.2636, 127.0286),
    "성남": (37.4449, 127.1389),
    "고양": (37.6584, 126.8320),
    "파주": (37.7600, 126.7800),
    "부산": (35.1796, 129.0756),
    "대구": (35.8714, 128.6014),
    "대전": (36.3504, 127.3845),
    "광주": (35.1595, 126.8526),
    "울산": (35.5384, 129.3114),
    "제주": (33.4996, 126.5312),
    # English weather prompts now reach the weather path, so the same
    # cities need ASCII keys too — otherwise _geocode() is handed a whole
    # English sentence and the lookup fails.
    # ponytail: alias table only; an English city that isn't listed still
    # falls through to _geocode() on the raw sentence and fails. Add a
    # proper place-name extractor only if non-Korean cities get asked for.
    "seoul": (37.5665, 126.9780),
    "incheon": (37.4563, 126.7052),
    "gimpo": (37.6153, 126.7156),
    "suwon": (37.2636, 127.0286),
    "seongnam": (37.4449, 127.1389),
    "goyang": (37.6584, 126.8320),
    "paju": (37.7600, 126.7800),
    "busan": (35.1796, 129.0756),
    "daegu": (35.8714, 128.6014),
    "daejeon": (36.3504, 127.3845),
    "gwangju": (35.1595, 126.8526),
    "ulsan": (35.5384, 129.3114),
    "jeju": (33.4996, 126.5312),
}

OFFICIAL_DOMAIN_HINTS = (
    (("openvino",), "site:docs.openvino.ai OR site:intel.com"),
    (("rhel", "red hat", "redhat"), "site:docs.redhat.com OR site:redhat.com OR site:developers.redhat.com"),
    (("openshift",), "site:docs.redhat.com OR site:redhat.com OR site:developers.redhat.com"),
    (("claude", "anthropic"), "site:docs.anthropic.com OR site:code.claude.com OR site:anthropic.com OR site:claude.com"),
    (("grok", "xai", "x.ai"), "site:docs.x.ai OR site:x.ai OR site:grok.com"),
    (("codex", "openai"), "site:developers.openai.com OR site:help.openai.com OR site:openai.com"),
    (("tesla", "model 3", "model y"), "site:tesla.com"),
)

OFFICIAL_SEARCH_TERMS = (
    (("openvino",), "OpenVINO official documentation"),
    (("rhel", "red hat", "redhat"), "RHEL 9 Red Hat official documentation"),
    (("openshift",), "OpenShift official storage documentation"),
    (("claude", "anthropic"), "Claude Code CLI reference official documentation site:code.claude.com"),
    (("grok", "xai", "x.ai"), "Grok Build official documentation"),
    (("codex", "openai"), "OpenAI Codex GPT-5-Codex official documentation"),
    (("tesla", "model 3", "model y"), "Tesla Model 3 official owner's manual Autopilot"),
)

GAME_PROFILES = {
    "lol": {
        "label": "League of Legends",
        "query_name": "롤",
        "aliases": ("리그 오브 레전드", "league of legends", "lol", "롤"),
        "official": (
            "leagueoflegends.com",
            "support-leagueoflegends.riotgames.com",
        ),
        "excluded_official": (
            "wildrift.leagueoflegends.com",
            "teamfighttactics.leagueoflegends.com",
        ),
        "trusted": (
            "wiki.leagueoflegends.com",
            "u.gg",
            "op.gg",
            "fow.lol",
            "lol.ps",
            "namu.wiki",
        ),
    },
    "arknights": {
        "label": "Arknights",
        "query_name": "명일방주",
        "aliases": ("명일방주", "arknights", "아크나이츠"),
        "official": ("arknights.global",),
        "trusted": (
            "arknights.wiki.gg",
            "gamepress.gg",
            "arknights.kr",
            "namu.wiki",
            "arca.live",
        ),
    },
    "genshin": {
        "label": "Genshin Impact",
        "query_name": "원신",
        "aliases": ("원신", "genshin impact", "genshin"),
        "official": (
            "genshin.hoyoverse.com",
            "hoyolab.com",
            "wiki.hoyolab.com",
        ),
        "shared_official": ("hoyolab.com",),
        "trusted": (
            "genshin-impact.fandom.com",
            "game8.co",
            "namu.wiki",
            "arca.live",
        ),
    },
    "zzz": {
        "label": "Zenless Zone Zero",
        "query_name": "젠레스 존 제로",
        "aliases": (
            "젠레스 존 제로",
            "젠레스존제로",
            "젤레스존제로",
            "zenless zone zero",
            "zzz",
        ),
        "official": (
            "zenless.hoyoverse.com",
            "hoyolab.com",
        ),
        "shared_official": ("hoyolab.com",),
        "trusted": (
            "zzz.wiki.gg",
            "zenless-zone-zero.fandom.com",
            "game8.co",
            "namu.wiki",
            "arca.live",
        ),
    },
    "palworld": {
        "label": "Palworld",
        "query_name": "팰월드",
        "aliases": ("팰월드", "펠월드", "palworld"),
        "official": (
            "news.palworldgame.com",
            "tech.palworldgame.com",
            "palworldgame.com",
            "pocketpair.jp",
        ),
        "trusted": (
            "palworld.wiki.gg",
            "game8.co",
            "palworld.gg",
            "op.gg",
            "namu.wiki",
            "store.steampowered.com",
            "steamcommunity.com",
        ),
    },
    "minecraft_apotheosis": {
        "label": "Minecraft / Apotheosis",
        "query_name": "Apotheosis Minecraft",
        "aliases": (
            "apotheosis",
            "아포테오시스",
            "apothic spawners",
            "apothic-spawners",
        ),
        "official": ("curseforge.com", "github.com"),
        "official_paths": (
            "/minecraft/mc-mods/apotheosis",
            "/minecraft/mc-mods/apothic-spawners",
            "/shadows-of-fire/apotheosis",
            "/shadows-of-fire/apothic-spawners",
        ),
        "trusted": (
            "wiki.siriusmc.net",
            "minecraft-apotheosis-mod.fandom.com",
            "minecraft-guides.com",
            "allthemods.github.io",
            "minecraft.wiki",
            "namu.wiki",
            "reddit.com",
        ),
        "search_language": "en-US",
    },
}

GAME_OFFICIAL_PATTERNS = (
    "공식",
    "official",
    "패치",
    "업데이트",
    "공지",
    "점검",
    "이벤트",
    "배너",
    "가챠",
    "확률",
    "쿠폰",
    "보상",
    "코드",
    "버전",
    "patch",
    "update",
    "event",
    "banner",
    "maintenance",
    "version",
)

GAME_GUIDE_PATTERNS = (
    "공략",
    "티어",
    "빌드",
    "조합",
    "파티",
    "팀",
    "육성",
    "캐릭터",
    "오퍼레이터",
    "에이전트",
    "방부",
    "성유물",
    "무기",
    "장비",
    "스킬",
    "재료",
    "파밍",
    "드롭",
    "배합",
    "교배",
    "패시브",
    "만드는법",
    "만드는 법",
    "몬스터팜",
    "몹팜",
    "monster farm",
    "mob farm",
    "farm",
    "farms",
    "guide",
    "tier",
    "build",
    "team",
    "party",
    "artifact",
    "weapon",
    "breeding",
    "passive",
)

WEATHER_CODES = {
    0: "맑음",
    1: "대체로 맑음",
    2: "부분적으로 흐림",
    3: "흐림",
    45: "안개",
    48: "착빙성 안개",
    51: "약한 이슬비",
    53: "이슬비",
    55: "강한 이슬비",
    56: "약한 어는 이슬비",
    57: "강한 어는 이슬비",
    61: "약한 비",
    63: "비",
    65: "강한 비",
    66: "약한 어는 비",
    67: "강한 어는 비",
    71: "약한 눈",
    73: "눈",
    75: "강한 눈",
    77: "눈알갱이",
    80: "약한 소나기",
    81: "소나기",
    82: "강한 소나기",
    85: "약한 눈 소나기",
    86: "강한 눈 소나기",
    95: "뇌우",
    96: "우박을 동반한 뇌우",
    99: "강한 우박을 동반한 뇌우",
}


# A game name alone is not a live-info signal — "롤이 몇 년도에 출시됐어?"
# is trivia, not a request for current patch/server state. Require one of
# these alongside the game name before treating it as a live query; a
# prompt that already matches LIVE_PATTERNS (최신, 최근, ...) doesn't need
# this check at all.
GAME_INTENT_PATTERNS = (
    "패치",
    "업데이트",
    "공지",
    "이벤트",
    "점검",
    "서버 상태",
    "다운로드",
    "랭크",
    "시즌",
    "신규",
    "밸런스",
    "너프",
    "버프",
    "patch",
    "update",
    "news",
    "maintenance",
    "event",
    "season",
    "release notes",
    "changelog",
)


# "오늘 날짜 형식(YYYY-MM-DD)을 설명해줘" asks about a date FORMAT, not
# today's date, so a date/time literal alone must not make it live when a
# format word is present. Other live signals (날씨, 뉴스, ...) still count.
FORMAT_WORDS = ("형식", "포맷", "표기", "format")


def is_live_query(prompt: str) -> bool:
    value = prompt.lower()

    hits = [pattern for pattern in LIVE_PATTERNS if pattern in value]
    if any(word in value for word in FORMAT_WORDS):
        hits = [pattern for pattern in hits if pattern not in TIME_PATTERNS]
    if hits:
        return True

    if detect_game_profile(prompt) is not None and any(
        pattern in value for pattern in GAME_INTENT_PATTERNS
    ):
        return True

    return False


def detect_game_profile(prompt: str):
    value = prompt.lower()
    candidates = []
    for name, profile in GAME_PROFILES.items():
        for alias in profile["aliases"]:
            if alias in value:
                candidates.append((len(alias), name))
    if not candidates:
        return None
    return max(candidates)[1]


def is_weather_query(prompt: str) -> bool:
    value = prompt.lower()
    return any(pattern in value for pattern in WEATHER_PATTERNS)


def is_time_query(prompt: str) -> bool:
    value = prompt.lower()
    return any(pattern in value for pattern in TIME_PATTERNS)


def _now_text() -> str:
    return datetime.now(KST).strftime(
        "%Y-%m-%d %H:%M:%S KST"
    )


def _request_json(
    url: str,
    timeout: int,
) -> Dict:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "IntelLocalAIRouter/0.11.0",
        },
        method="GET",
    )

    with urllib.request.urlopen(
        request,
        timeout=timeout,
    ) as response:
        return json.loads(
            response.read().decode(
                "utf-8",
                errors="replace",
            )
        )


def _clean_text(value: str) -> str:
    value = html.unescape(value or "")
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _build_search_query(prompt: str) -> str:
    query = re.sub(
        r"웹\s*검색|검색해|검색해서|찾아줘|찾아 주세요|알려줘|알려 주세요|"
        r"공식\s*문서|공식\s*링크|출처|링크",
        " ",
        prompt,
        flags=re.IGNORECASE,
    )
    query = re.sub(r"\s+", " ", query).strip()
    query = re.sub(r"\s+로(?=\s|$)", " ", query)
    query = re.sub(r"[.,!?]+", " ", query)
    query = re.sub(r"\s+", " ", query).strip()

    official_request = any(
        marker in prompt.lower()
        for marker in ("공식", "official", "출처")
    )

    if official_request:
        for keywords, domain_hint in OFFICIAL_DOMAIN_HINTS:
            if any(
                keyword in prompt.lower()
                for keyword in keywords
            ):
                query += " official documentation " + domain_hint
                break

    return query or prompt


def _game_query_text(prompt: str, profile) -> str:
    query = _build_search_query(prompt)
    for alias in profile["aliases"]:
        query = re.sub(
            re.escape(alias),
            " ",
            query,
            flags=re.IGNORECASE,
        )
    query = re.sub(
        r"공식\s*/\s*위키|공식|위키|출처|간결하게|간단히|자세하게|"
        r"알려줘|알려 주세요|답해줘|답해 주세요|찾아줘|찾아 주세요|"
        r"패치\s*버전과\s*지역을\s*구분해줘|"
        r"패치\s*버전과\s*지역을\s*써줘|지역을\s*써줘|"
        r"글로벌\s*서버\s*기준으로\s*답해줘|기준으로\s*답해줘|"
        r"구분해줘|구분해 주세요|5줄\s*이내(?:로)?|"
        r"만드는\s*법|몬스터\s*팜|몹\s*팜|및",
        " ",
        query,
        flags=re.IGNORECASE,
    )
    query = re.sub(r"\s+", " ", query).strip()
    query = re.sub(r"\s+로(?=\s|$)", " ", query)
    query = re.sub(r"[.,!?]+", " ", query)
    query = re.sub(r"\s+", " ", query).strip()
    return query or profile["query_name"]


def _build_game_search_queries(prompt: str, game: str):
    profile = GAME_PROFILES[game]
    base = _game_query_text(prompt, profile)
    official_site = f" site:{profile['official'][0]}"
    value = prompt.lower()
    official_intent = any(
        pattern in value
        for pattern in GAME_OFFICIAL_PATTERNS
    )
    guide_intent = any(
        pattern in value
        for pattern in GAME_GUIDE_PATTERNS
    )
    queries = []

    if game == "minecraft_apotheosis":
        if official_intent or not guide_intent:
            queries.append(
                (
                    "Apotheosis Minecraft mod official"
                    + official_site,
                    "official",
                )
            )
            queries.append(
                (
                    "Apothic Spawners Minecraft official"
                    + official_site,
                    "official",
                )
            )
        if guide_intent or not official_intent:
            queries.append(
                (
                    "Apotheosis Minecraft mod guide spawner",
                    "trusted",
                )
            )
        return queries

    if official_intent or not guide_intent:
        queries.append(
            (
                f"{profile['query_name']} {base} "
                f"{profile['label']} 공식 패치 공지 업데이트"
                f"{official_site}",
                "official",
            )
        )
        queries.append(
            (
                f"{profile['label']} latest official news"
                f"{official_site}",
                "official",
            )
        )

    if guide_intent or not official_intent:
        queries.append(
            (
                f"{profile['query_name']} {base} "
                f"공략 티어 조합 육성 데이터베이스",
                "trusted",
            )
        )

    return queries or [
        (
            f"{profile['query_name']} {base}",
            "general",
        )
    ]


def _location_from_prompt(prompt: str):
    for name, coordinates in sorted(
        LOCATION_ALIASES.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if name in prompt.lower():
            return name, coordinates

    cleaned = re.sub(
        r"오늘|내일|이번\s*주|주간|현재|실시간|날씨|기온|강수|확률|알려줘|알려주세요|정보|예보|태풍|기상",
        " ",
        prompt,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"[^0-9A-Za-z가-힣 ]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or "서울", None


def _geocode(
    location: str,
):
    params = urllib.parse.urlencode(
        {
            "name": location,
            "count": 1,
            "language": "ko",
            "format": "json",
        }
    )
    data = _request_json(
        "https://geocoding-api.open-meteo.com/v1/search?"
        + params,
        WEATHER_TIMEOUT,
    )
    results = data.get("results") or []
    if not results:
        return None
    item = results[0]
    return (
        item.get("name") or location,
        item.get("country") or "",
        float(item["latitude"]),
        float(item["longitude"]),
    )


def _weather_context(
    prompt: str,
):
    location, coordinates = _location_from_prompt(prompt)
    if coordinates:
        place = (location, "대한민국", *coordinates)
    else:
        place = _geocode(location)

    if not place:
        raise RuntimeError(
            "위치를 확인할 수 없습니다. 도시나 지역명을 포함해 주세요."
        )

    name, country, latitude, longitude = place
    params = urllib.parse.urlencode(
        {
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "temperature_2m,relative_humidity_2m,"
                "weather_code,wind_speed_10m"
            ),
            "daily": (
                "temperature_2m_max,temperature_2m_min,"
                "precipitation_probability_max,weather_code,"
                "wind_speed_10m_max"
            ),
            "forecast_days": 7,
            "timezone": "Asia/Seoul",
        }
    )
    url = "https://api.open-meteo.com/v1/forecast?" + params
    data = _request_json(url, WEATHER_TIMEOUT)
    current = data.get("current") or {}
    daily = data.get("daily") or {}
    dates = daily.get("time") or []
    maximums = daily.get("temperature_2m_max") or []
    minimums = daily.get("temperature_2m_min") or []
    rain = daily.get("precipitation_probability_max") or []
    codes = daily.get("weather_code") or []
    winds = daily.get("wind_speed_10m_max") or []

    lines = [
        "[LIVE_WEATHER_DATA]",
        "조회 시각: " + _now_text(),
        f"위치: {name}, {country}",
        (
            "현재: "
            + str(current.get("temperature_2m", "확인 불가"))
            + "°C, "
            + WEATHER_CODES.get(
                current.get("weather_code"),
                "상태 확인 불가",
            )
            + ", 습도 "
            + str(current.get("relative_humidity_2m", "확인 불가"))
            + "%, 바람 "
            + str(current.get("wind_speed_10m", "확인 불가"))
            + " km/h"
        ),
        "7일 예보:",
    ]

    for index, date in enumerate(dates):
        lines.append(
            f"- {date}: "
            f"{minimums[index]}~{maximums[index]}°C, "
            f"{WEATHER_CODES.get(codes[index], '상태 확인 불가')}, "
            f"강수확률 {rain[index]}%, "
            f"최대풍속 {winds[index]} km/h"
        )

    return {
        "kind": "weather",
        "ok": True,
        "context": "\n".join(lines),
        "sources": [
            {
                "title": "Open-Meteo Forecast API",
                "url": url,
            }
        ],
    }


def _time_context():
    now = _now_text()
    return {
        "kind": "time",
        "ok": True,
        "context": (
            "[LIVE_CLOCK_DATA]\n"
            "조회 시각: "
            + now
            + "\n현재 한국 표준시: "
            + now
        ),
        "sources": [
            {
                "title": "Local server clock",
                "url": "local://clock",
            }
        ],
    }


def _search_once(
    search_query: str,
    prompt: str,
    game: bool = False,
    official: bool = False,
    language: str = "all",
):
    params = {
        "q": search_query,
        "format": "json",
        "language": language,
        "safesearch": 1,
        "categories": "general",
        "pageno": 1,
    }
    if game or official:
        # SearXNG engine shortcuts are stable here; full names can be parsed
        # as an engine group and return unrelated results on this instance.
        params["engines"] = "bi"

    if not game and any(
        pattern in prompt
        for pattern in ("뉴스", "속보", "오늘", "최근", "최신")
    ):
        params["time_range"] = "day"

    def make_url():
        return (
            SEARXNG_URL
            + "/search?"
            + urllib.parse.urlencode(params)
        )

    query_url = make_url()
    data = _request_json(query_url, SEARCH_TIMEOUT)
    results = data.get("results") or []

    if not results and "time_range" in params:
        params.pop("time_range")
        query_url = make_url()
        data = _request_json(query_url, SEARCH_TIMEOUT)
        results = data.get("results") or []

    return results, query_url


def _domain_matches(host: str, domain: str) -> bool:
    domain = domain.lower().removeprefix("www.")
    return host == domain or host.endswith("." + domain)


def _profile_official_match(url: str, profile) -> bool:
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    if not any(
        _domain_matches(host, domain)
        for domain in profile.get("official", ())
    ):
        return False

    paths = profile.get("official_paths", ())
    if not paths:
        return True

    target = parsed.path.rstrip("/").lower()
    return any(
        target == path.lower()
        or target.startswith(path.lower().rstrip("/") + "/")
        for path in paths
    )


def _official_domains(prompt: str):
    value = prompt.lower()
    for keywords, domain_hint in OFFICIAL_DOMAIN_HINTS:
        if any(keyword in value for keyword in keywords):
            return tuple(
                re.findall(
                    r"site:([a-z0-9.-]+)",
                    domain_hint,
                )
            )
    return ()


def _official_search_query(prompt: str):
    value = prompt.lower()
    for keywords, query in OFFICIAL_SEARCH_TERMS:
        if any(keyword in value for keyword in keywords):
            return query
    return ""


def _source_quality(url: str, profile=None) -> str:
    if not profile:
        return "general"
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    if _profile_official_match(url, profile):
        if "hoyolab.com" in host and not (
            "/official/" in urllib.parse.urlparse(url).path
            or host.startswith("wiki.")
        ):
            return "community"
        return "official"
    if any(
        _domain_matches(host, domain)
        for domain in profile["trusted"]
    ):
        if (
            "wiki" in host
            or "fandom" in host
            or host == "namu.wiki"
        ):
            return "wiki"
        return "community"
    return "general"


def _game_result_relevant(
    url: str,
    title: str,
    profile,
) -> bool:
    host = (urllib.parse.urlparse(url).hostname or "").lower()

    if any(
        _domain_matches(host, domain)
        for domain in profile.get("excluded_official", ())
    ):
        return False

    official_host = any(
        _domain_matches(host, domain)
        for domain in profile.get("official", ())
    )
    official = _profile_official_match(url, profile)
    if official_host and profile.get("official_paths") and not official:
        return False
    shared_official = any(
        _domain_matches(host, domain)
        for domain in profile.get("shared_official", ())
    )

    if official and not shared_official:
        return True

    # Precision first: trusted sites can mention a game in a generic snippet;
    # require the title or URL to identify the requested game.
    title_url = re.sub(
        r"[_-]+",
        " ",
        f"{title} {url}".lower(),
    )
    return any(
        alias.lower() in title_url
        for alias in profile["aliases"]
    )


def _canonical_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path.rstrip("/"),
            parsed.query,
            "",
        )
    )


def _search_context(
    prompt: str,
):
    game = detect_game_profile(prompt)
    game_official_intent = any(
        pattern in prompt.lower()
        for pattern in GAME_OFFICIAL_PATTERNS
    )
    game_guide_intent = any(
        pattern in prompt.lower()
        for pattern in GAME_GUIDE_PATTERNS
    )
    official_request = any(
        marker in prompt.lower()
        for marker in ("공식", "official", "출처")
    )
    official_domains = (
        _official_domains(prompt)
        if official_request
        else ()
    )
    if game:
        plans = _build_game_search_queries(prompt, game)
        profile = GAME_PROFILES[game]
    elif official_domains:
        query = _official_search_query(prompt)
        plans = [
            (
                query or _build_search_query(prompt),
                "official",
            )
        ]
        profile = None
    else:
        plans = [(_build_search_query(prompt), "general")]
        profile = None

    collected = []
    seen = set()
    query_urls = []
    errors = []

    def run_plan(plan):
        search_query, plan_quality = plan
        try:
            return _search_once(
                search_query,
                prompt,
                game=bool(game),
                official=(
                    bool(official_domains)
                    or plan_quality == "official"
                ),
                language=(
                    profile.get("search_language", "all")
                    if game
                    else "all"
                ),
            ), ""
        except Exception as exc:
            return None, str(exc)

    if len(plans) > 1:
        # ponytail: cap at two workers to match the current two-query plan.
        with ThreadPoolExecutor(max_workers=2) as pool:
            plan_results = list(pool.map(run_plan, plans))
    else:
        plan_results = [run_plan(plans[0])]

    for (search_query, plan_quality), (plan_result, error) in zip(
        plans,
        plan_results,
    ):
        if error:
            errors.append(error)
            continue

        results, query_url = plan_result
        query_urls.append(query_url)
        result_limit = 20 if official_domains else 12 if game else 5
        for rank, item in enumerate(results[:result_limit]):
            title = _clean_text(item.get("title", ""))
            url = item.get("url", "")
            if not url:
                continue
            snippet = _clean_text(
                item.get("content")
                or item.get("snippet")
                or ""
            )
            key = _canonical_url(url)
            if key in seen:
                continue
            seen.add(key)
            if official_domains:
                host = (
                    urllib.parse.urlparse(url).hostname
                    or ""
                ).lower()
                if not any(
                    _domain_matches(host, domain)
                    for domain in official_domains
                ):
                    continue
            if game and not _game_result_relevant(
                url=url,
                title=title,
                profile=profile,
            ):
                continue
            quality = _source_quality(url, profile)
            if official_domains:
                quality = "official"
            if game and quality == "general":
                continue
            collected.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "quality": quality,
                    "plan_quality": plan_quality,
                    "rank": rank,
                }
            )

    if game and game_official_intent:
        official_results = [
            item
            for item in collected
            if item["quality"] == "official"
        ]
        if official_results:
            collected = official_results

    if game and game_guide_intent:
        guide_results = [
            item
            for item in collected
            if item["quality"] in {"wiki", "community"}
        ]
        if guide_results:
            collected = guide_results

    if not collected:
        detail = errors[-1] if errors else "검색 결과가 없습니다."
        raise RuntimeError(detail)

    quality_order = {
        "official": 0,
        "wiki": 1,
        "community": 2,
        "general": 3,
    }
    collected.sort(
        key=lambda item: (
            quality_order.get(item["quality"], 9),
            item["rank"],
        )
    )
    limit = 8 if game else 5
    collected = collected[:limit]

    if game:
        lines = [
            "[GAME_SEARCH_DATA]",
            "게임: " + profile["label"],
            "게임 키: " + game,
            "조회 시각: " + _now_text(),
            "검색어: " + prompt,
            "출처 등급: official=공식, wiki=위키/게임 DB, community=통계/공략, general=기타",
            "게임 데이터 규칙: 패치·공지·버전·확률은 공식 출처를 우선하고, 공략·티어·육성·DB는 위키/통계 출처를 사용한다. 서버·지역·플랫폼·버전을 섞지 말고 출처가 충돌하면 날짜와 차이를 명시한다.",
        ]
    else:
        lines = [
            "[LIVE_WEB_SEARCH_DATA]",
            "조회 시각: " + _now_text(),
            "검색어: " + prompt,
        ]

    lines.append("실제 검색 질의:")
    lines.extend(
        "- " + search_query
        for search_query, _ in plans
    )
    sources = []

    for index, item in enumerate(collected, 1):
        lines.extend(
            [
                f"{index}. {item['title']}",
                "품질: " + item["quality"],
                "URL: " + item["url"],
                "발췌: " + item["snippet"][:600],
            ]
        )
        sources.append(
            {
                "title": item["title"],
                "url": item["url"],
                "quality": item["quality"],
            }
        )

    result = {
        "kind": "game-search" if game else "search",
        "ok": True,
        "context": "\n".join(lines),
        "sources": sources,
        "query_url": query_urls[0] if query_urls else "",
        "query_urls": query_urls,
    }
    if game:
        result["game"] = game
        result["game_label"] = profile["label"]
    return result


def fetch_live_context(
    prompt: str,
):
    started = time.perf_counter()
    game = detect_game_profile(prompt)
    try:
        if is_time_query(prompt):
            result = _time_context()
        elif is_weather_query(prompt):
            result = _weather_context(prompt)
            if any(
                pattern in prompt
                for pattern in ("태풍", "기상청", "경보", "특보")
            ):
                try:
                    search = _search_context(prompt)
                    result["context"] += (
                        "\n\n"
                        + search["context"]
                    )
                    result["sources"].extend(
                        search["sources"]
                    )
                    result["kind"] = "weather+search"
                except Exception as exc:
                    result["search_error"] = str(exc)
        else:
            result = _search_context(prompt)
        result["elapsed"] = time.perf_counter() - started
        return result
    except Exception as exc:
        result = {
            "kind": (
                "game-search"
                if game
                else "weather" if is_weather_query(prompt) else "search"
            ),
            "ok": False,
            "context": "",
            "sources": [],
            "error": str(exc),
            "elapsed": time.perf_counter() - started,
        }
        if game:
            result["game"] = game
            result["game_label"] = GAME_PROFILES[game]["label"]
        return result
