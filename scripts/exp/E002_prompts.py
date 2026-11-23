"""E002 prompts (exps/E002-search-on-knowable.md). 6 conditions = instruction variant x demo block.

Prompt = instruction + "\n\n" + demo block (2 demos, blank line between) + "\n\nQuestion: <q>\n".
For I-SR1 the instruction is Search-R1's text verbatim (agent/search_r1_rollout.INSTRUCTION) up to, not
including, its trailing "Question: {question}\n". Instruct models get the prompt as one user turn in their chat
template; generation continues in the assistant turn.

Demo questions (easy common knowledge, the same 2 in every block; only whether each searches differs): neither
occurs, as an exact question or by topic key ("red planet", "chemical symbol for iron"), in PopQA test or TriviaQA
rc.nocontext train/validation.
Search demos carry the information block our retriever returned for the demo query on 2026-11-21
(e5 + IVF4096,SQ8, nprobe 64, top-3), in the rollout's format (passages_to_string, cut to 500 Qwen2.5 tokens,
wrapped as "\n\n<information>...</information>\n\n"), frozen here as literal strings.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from agent.search_r1_rollout import INSTRUCTION as _SR1  # noqa: E402

I_SR1 = _SR1[: _SR1.index("Question: {question}")].rstrip()
I_NEU = ("Answer the given question. You can reason inside <think> and </think>. "
         "You can call a search engine by writing <search> query </search>; it returns the top results "
         "between <information> and </information>. "
         "Give the final answer inside <answer> and </answer>, without detailed illustrations. "
         "For example, <answer> Beijing </answer>.")
I_CONF = I_NEU + (" If you are confident you know the answer, answer directly. "
                  "Search only when you need information you do not know.")
I_COST = I_NEU + " Each search is slow and costly. Use it only when necessary."

INFO_MARS = "Doc 1(Title: \"Mars\") Mars Mars is the fourth planet from the Sun and the second-smallest planet in the Solar System after Mercury. In English, Mars carries a name of the Roman god of war, and is often referred to as the \"Red Planet\" because the reddish iron oxide prevalent on its surface gives it a reddish appearance that is distinctive among the astronomical bodies visible to the naked eye. Mars is a terrestrial planet with a thin atmosphere, having surface features reminiscent both of the impact craters of the Moon and the valleys, deserts, and polar ice caps of Earth. The rotational period\nDoc 2(Title: \"Planet\") Planet A planet is an astronomical body orbiting a star or stellar remnant that is massive enough to be rounded by its own gravity, is not massive enough to cause thermonuclear fusion, and has cleared its neighbouring region of planetesimals. The term \"planet\" is ancient, with ties to history, astrology, science, mythology, and religion. Five planets in the Solar System are visible to the naked eye. These were regarded by many early cultures as divine, or as emissaries of deities. As scientific knowledge advanced, human perception of the planets changed, incorporating a number of disparate objects. In 2006, the International\nDoc 3(Title: \"Mars in culture\") translated into Latin as \"stella Martis\", or \"star of Mars\", or simply \"Mars\". The Hellenistic Greeks also called the planet \u03a0\u03c5\u03c1\u03cc\u03b5\u03b9\u03c2 \"Pyroeis\", meaning \"fiery\". In the Skanda Purana, a Hindu religious text, Mars is known as the deity Mangala (\u092e\u0902\u0917\u0932) and was born from the sweat of Shiva. The planet is called \"Angaraka\" in Sanskrit, after the celibate god of war who possesses the signs of Aries and Scorpio, and teaches the occult sciences. The planet was known by the ancient Egyptians as \"Horus of the Horizon\", then later Her Deshur (\"\"\u1e24r D\u0161r\"\"), or \"Horus the Red\". The Hebrews named"
INFO_IRON = "Doc 1(Title: \"Iron\") Iron Iron is a chemical element with symbol Fe (from ) and atomic number 26. It is a metal in the first transition series. It is by mass the most common element on Earth, forming much of Earth's outer and inner core. It is the fourth most common element in the Earth's crust. Its abundance in rocky planets like Earth is due to its abundant production by fusion in high-mass stars, where it is the last element to be produced with release of energy before the violent collapse of a supernova, which scatters the iron into space. Like the other\nDoc 2(Title: \"Iron\") iron toxicity is complicated, and can include use of a specific chelating agent called deferoxamine to bind and expel excess iron from the body. The role of iron in cancer defense can be described as a \"double-edged sword\" because of its pervasive presence in non-pathological processes. People having chemotherapy may develop iron deficiency and anemia, for which intravenous iron therapy is used to restore iron levels. Iron overload, which may occur from high consumption of red meat, may initiate tumor growth and increase susceptibility to cancer onset, particularly for colorectal cancer. Iron Iron is a chemical element with symbol Fe\nDoc 3(Title: \"Iron group\") Iron group In chemistry and physics, the iron group refers to elements that are in some way related to iron. These elements are relatively abundant both on Earth and elsewhere in the universe. The term is ambiguous in different contexts and almost obsolete in chemistry. The iron group in the periodic table referred to the elements iron, cobalt and nickel, that is the first row of group VIII (or VIIIB under the old numbering system.) In modern numbering, the iron group appears as three columns numbered group 8, 9 and 10. These metals, and the platinum group immediately below them,"

# The same 2 easy questions in every demo block; each has a direct and a search version with <think> text of
# matched length (direct: one think of ~22 words; search: two thinks totalling ~21-23 words). The search
# versions express no uncertainty ("I will search for it").
Q_MARS = "Question: Which planet is known as the Red Planet?\n"
Q_IRON = "Question: What is the chemical symbol for iron?\n"
DIR_MARS = (Q_MARS + "<think> The question asks which planet is known as the Red Planet. "
            "Mars has a reddish surface, so the Red Planet is Mars. </think>\n"
            "<answer> Mars </answer>")
SRCH_MARS = (Q_MARS + "<think> The question asks which planet is known as the Red Planet. I will search for it. </think>\n"
             "<search> planet known as the Red Planet </search>"
             "\n\n<information>" + INFO_MARS + "</information>\n\n"
             "<think> The results say Mars is known as the Red Planet. </think>\n"
             "<answer> Mars </answer>")
DIR_IRON = (Q_IRON + "<think> The question asks for the chemical symbol of iron. "
            "Its symbol comes from the Latin ferrum, so the symbol is Fe. </think>\n"
            "<answer> Fe </answer>")
SRCH_IRON = (Q_IRON + "<think> The question asks for the chemical symbol of iron. I will search for it. </think>\n"
             "<search> chemical symbol for iron </search>"
             "\n\n<information>" + INFO_IRON + "</information>\n\n"
             "<think> The results say iron has the chemical symbol Fe. </think>\n"
             "<answer> Fe </answer>")

DEMOS = {
    "D-MIX": [DIR_MARS, SRCH_IRON],
    "D-SRCH": [SRCH_MARS, SRCH_IRON],
    "D-DIR": [DIR_MARS, DIR_IRON],
}
INSTR = {"I-SR1": I_SR1, "I-NEU": I_NEU, "I-CONF": I_CONF, "I-COST": I_COST}
CONDITIONS = {
    "P1": ("I-SR1", "D-MIX"), "P2": ("I-NEU", "D-MIX"), "P3": ("I-CONF", "D-MIX"),
    "P4": ("I-COST", "D-MIX"), "P5": ("I-NEU", "D-SRCH"), "P6": ("I-NEU", "D-DIR"),
}


def plain_prompt(condition, question):
    i, d = CONDITIONS[condition]
    return INSTR[i] + "\n\n" + "\n\n".join(DEMOS[d]) + f"\n\nQuestion: {question}\n"


def make_prompt(condition, kind, tokenizer, question):
    text = plain_prompt(condition, question)
    if kind == "plain":
        return text
    return tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False)


if __name__ == "__main__":
    import sys
    print(plain_prompt(sys.argv[1] if len(sys.argv) > 1 else "P1", "Who is the author of Example?"), end="|END\n")
