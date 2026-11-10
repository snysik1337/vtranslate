"""Pronunciation fixes for the Piper voices, written for vtranslate.

Piper reads Russian through espeak-ng, which never guesses a missing ё and
stresses a few common words wrongly. Translations usually leave ё out, so
"пойдет" was read with е instead of ё. Each entry below was checked against the
voices: only words they get wrong are listed, because marking a word Piper
already reads right would drop its own pronunciation rules (конечно → "конешно").

Words whose stress depends on meaning (воды, сети, узнаете, все) are left alone:
a list can't tell which one is meant.
"""
from __future__ import annotations

import re

ACCENT = "\u0301"  # combining acute after a vowel: Piper stresses that vowel
VOWELS = "аеёиоуыэюя"
# "+" goes before the stressed vowel.
STRESS = """
п+еред чт+о-то чт+о-нибудь чт+о-либо п+омните х+остов х+осты включ+ат подключ+ит
подключ+ат отключ+ит догов+ора т+ортов щав+ель оптов+ый др+айвера др+айверов х+акинг
х+акинга м+аркетинг прот+окол занят+а
"""
# Common words that always have ё (берёт is left out: "берет" is also a beret).
YO = """
ещё её всё-таки ёмкость ёмкости идёт пойдёт придёт найдёт начнёт живёт ждёт даёт
создаёт встаёт продаёт передаёт остаётся ведёт несёт растёт поёт пьёт зовёт перейдёт
пройдёт войдёт произойдёт подойдёт уйдёт идём пойдём создаём даём начнём найдём поймём
пройдём перейдём разберёмся ждём живём берём ведём идёте пойдёте найдёте создаёте даёте
начнёте ведёте берёте ждёте счёт счёта счёте счёту учёт учёта учёте отчёт отчёта отчёты
отчётов расчёт расчёта расчёты зачёт учётная учётной учётную учётные учётных лёгкий
лёгкая лёгкое лёгкие лёгкого лёгкой тёмный тёмная тёмное тёмные тёмного тёмной чёрный
чёрная чёрное чёрные чёрного чёрной жёлтый жёлтая жёлтое жёлтые зелёный зелёная зелёное
зелёные твёрдый твёрдая твёрдое твёрдые твёрдого тёплый тёплая тёплое тёплые жёсткий
жёсткого жёсткая жёсткие жёстком приём приёма приёме объём объёма объёмы подъём вперёд
причём ребёнок ребёнка актёр шофёр партнёр партнёра партнёры бельё жильё моё твоё своё
ёлка мёд лёд нёс вёл шёл пришёл ушёл нашёл пошёл зашёл вошёл определённый определённая
определённое определённые определённого определённой обновлённый обновлённая обновлённое
обновлённые подключённый подключённая подключённое подключённые упрощённый упрощённая
защищённый защищённая защищённое защищённые полёт самолёт самолёта вертолёт отдалённый
удалённый удалённая удалённое удалённые удалённого удалённой напряжённый объединённый
объединённая объединённые сохранённый сохранённые четырёх трёх трёхмерный четырёхзначный
"""
# A stressed word joined to these is written as one word ("что́то"), or Piper
# stresses the particle too.
PARTICLES = ("-то", "-нибудь", "-либо")


def _marked(word: str) -> str:
    i = word.index("+")
    fixed = word[:i] + word[i + 1] + ACCENT + word[i + 2:]
    for particle in PARTICLES:
        if fixed.endswith(particle):
            fixed = fixed[: -len(particle)] + particle[1:]
    return fixed


FIXES = {word.replace("+", ""): _marked(word) for word in STRESS.split()}
YO_FIXES = {word.replace("ё", "е"): word for word in YO.split()}
FIXES.update(YO_FIXES)
WORD = re.compile(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*")


def fix(text: str, marks: bool = True) -> str:
    """The text with the listed words spelled so Piper says them right.

    With marks=False only ё is restored, which is right for any reader.
    """
    table = FIXES if marks else YO_FIXES

    def replace(match: re.Match) -> str:
        word = match.group()
        fixed = table.get(word.lower())
        if fixed is None:
            return word
        if word.isupper() and len(word) > 1:
            return fixed.upper()
        return fixed[0].upper() + fixed[1:] if word[0].isupper() else fixed

    return WORD.sub(replace, text)
