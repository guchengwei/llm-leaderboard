import time

import wandb
import pandas as pd

from ....config_singleton import WandbConfigSingleton
from ....utils import read_wandb_table
from .controllability import controllability_dict

def evaluate():
    time.sleep(30)  # wait for upload dataset
    instance = WandbConfigSingleton.get_instance()
    run = instance.run

    input_task = "jaster_0shot"
    output_task = input_task + "_controllability"
    for table_suffix in ("", "_dev"):
        # get output table
        table_name = f"{input_task}_output_table{table_suffix}"
        output_df = read_wandb_table(run=run, table_name=table_name)
        # evaluate controllability
        output_df["metrics"] = output_df["task"].map(
            {k: v.__name__ for k, v in controllability_dict.items()}
        )
        output_df.dropna(subset=["metrics"], axis=0, inplace=True)
        output_df["score"] = output_df.apply(
            lambda x: controllability_dict[x["task"]](x["output"]) * 1, axis=1
        )
        # log tables
        table_dict = {
            f"{output_task}_output_table{table_suffix}": output_df,
        }
        if table_suffix == "":
            leaderboard_df = pd.pivot_table(
                data=output_df,
                values="score",
                index=["run_name", "model_name"],
                columns="dataset",
                aggfunc="mean",
            ).reset_index()
            leaderboard_df.columns = [
                "run_name",
                "model_name",
                output_task,
            ]
            table_dict.update(
                {
                    f"{output_task}_leaderboard_table": leaderboard_df,
                }
            )
        wandb.log(table_dict)

# ---------------------
# For controllability
# ---------------------

# mawps, mgsm
def is_all_digit(text: str) -> int:
    try:
        float(text)
        return 1
    except ValueError:
        return 0

# jmmlu, mmlu_prox_ja
def is_one_of_ABCD(text: str) -> int:
    return 1 if text in {"A", "B", "C", "D"} else 0

# JBLiMP
def is_a_b(text: str) -> int:
    return 1 if text in {"a", "b"} else 0

# jcommonsenseqa
def is_0_4(text: str) -> int:
    return 1 if text in {"0", "1", "2", "3", "4"} else 0

# kuci
def is_0_3(text: str) -> int:
    return 1 if text in {"0", "1", "2", "3"} else 0

# jcola, JCommonsenseMorality
def is_0_1(text: str) -> int:
    return 1 if text in {"0", "1"} else 0

# janli
def is_entailment2_format(text: str) -> int:
    return 1 if text in {"entailment", "non-entailment"} else 0

# jnli, jsick, jamp
def is_entailment3_format(text: str) -> int:
    return 1 if text in {"entailment", "contradiction", "neutral"} else 0

# jsem
def is_jsem_format(text: str) -> int:
    return 1 if text in {"yes", "no", "unknown", "undef"} else 0

# no_check
def no_check(text: str):
    return None

controllability_dict = {
    "aio": no_check,
    "alt-e-to-j": no_check,
    "alt-j-to-e": no_check,
    "commonsensemoralja": is_0_1,
    "jamp": is_entailment3_format,
    "janli": is_entailment2_format,
    "jblimp": is_a_b,
    "jcola-in-domain": is_0_1,
    "jcola-out-of-domain": is_0_1,
    "jcommonsenseqa": is_0_4,
    "jemhopqa": no_check,
    "jhumaneval": no_check,
    "jnli": is_entailment3_format,
    "jsem": is_jsem_format,
    "jsick": is_entailment3_format,
    "jsquad": no_check,
    "jmmlu": is_one_of_ABCD,
    "mmlu_prox_ja": is_one_of_ABCD,
    "mmlu_en": is_one_of_ABCD,
    "kuci": is_0_3,
    "mawps": is_all_digit,
    "mgsm": is_all_digit,
    "niilc": no_check,
}
