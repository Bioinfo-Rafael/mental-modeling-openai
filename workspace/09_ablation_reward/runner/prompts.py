"""Remove only target-history rewards; keep the frozen Exp.08 prefix verbatim."""
import re


def without_rewards(query):
    history, count = re.subn(r'\n  reward: [^\n]*', '', query.history_text, flags=re.MULTILINE)
    if count != query.history_end - query.history_start:
        raise ValueError('Expected exactly one reward line per history step')
    if query.user_prompt.count(query.history_text) != 1:
        raise ValueError('Expected exactly one target history')
    return query.user_prompt.replace(query.history_text, history, 1)
