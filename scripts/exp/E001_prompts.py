"""E001 prompts (exps/E001-popqa-closed-book-labels.md), revision 2 (2026-11-20): 3-shot, closed-book.

Prompt = INSTRUCTION, a blank line, three demonstrations separated by blank lines, a blank line, then
"Question: <q>\n". Base and RL checkpoints get this as PLAIN text. Instruct models get it as one user
turn in the model's chat template (add_generation_prompt=True).
The demo questions do not occur (exact or substring) in PopQA test or TriviaQA rc.nocontext validation.
"""

INSTRUCTION = ("Answer the given question. Think inside <think></think> if needed, "
               "then give the final answer inside <answer></answer>.")

DEMOS = [
    ("Who wrote the play Romeo and Juliet?", "William Shakespeare"),
    ("In what country is the city of Lyon?", "France"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
]


def _demo(q, a):
    return f"Question: {q}\n<think></think>\n<answer>{a}</answer>"


def plain_prompt(question):
    return INSTRUCTION + "\n\n" + "\n\n".join(_demo(q, a) for q, a in DEMOS) + f"\n\nQuestion: {question}\n"


def chat_prompt(tokenizer, question):
    return tokenizer.apply_chat_template([{"role": "user", "content": plain_prompt(question)}],
                                         add_generation_prompt=True, tokenize=False)


def make_prompt(kind, tokenizer, question):
    return plain_prompt(question) if kind == "plain" else chat_prompt(tokenizer, question)


D = "/data/erfan/search-agent-rl-data"
# model tag -> (HF repo id or local path, prompt type)
MODELS = {
    "qwen3b-base": ("Qwen/Qwen2.5-3B", "plain"),
    "qwen3b-instruct": ("Qwen/Qwen2.5-3B-Instruct", "chat"),
    "searchr1-3b": ("PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo", "plain"),
    # Card model 4 (L001 step 60, ..._20261019_183501/ckpt/global_step_60) was deleted before 2026-11-20.
    # Substitute: E1 arm A step 60 (same recipe; 100-step schedule, warmup 0.06). Pending Erfan's decision.
    "grpo-e1a-s60": (f"{D}/logs/grpo_qwen2.5-3b_full_E1A_em_20261118_102220/hf/"
                     "grpo_qwen2.5-3b_full_E1A_em_20261118_102220_step60", "plain"),
    "qwen7b-instruct": ("Qwen/Qwen2.5-7B-Instruct", "chat"),
}

if __name__ == "__main__":
    print(plain_prompt("Who is the author of Example?"), end="|END\n")
