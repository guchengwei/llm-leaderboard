import math
import re
from fuzzywuzzy import fuzz
from scipy.stats import pearsonr, spearmanr
from statistics import mean
from sacrebleu import BLEU
import textwrap
from jinja2 import Template
import ast
import subprocess
from pathlib import Path
from .sandbox_client import CodeExecutor, is_sandbox_running
import logging


logger = logging.getLogger(__name__)

#import bert_score
import shutil
from comet import download_model, load_from_checkpoint
from abc import ABC, abstractmethod



task_to_sub_category = {
    # 応用的言語性能
    "alt-e-to-j": "GLP_translation",
    "alt-j-to-e": "GLP_translation",
    "jsquad": "GLP_information_extraction",
    "humanities": "GLP_expression",
    "roleplay": "GLP_expression",
    "writing": "GLP_expression",
    
    # 推論能力
    "reasoning": "GLP_logical_reasoning",
    "mawps": "GLP_mathematical_reasoning",
    "mgsm": "GLP_mathematical_reasoning",
    "math": "GLP_mathematical_reasoning",
    "arc_agi_2": "GLP_abstract_reasoning",
    
    # 知識・質問応答
    "jcommonsenseqa": "GLP_general_knowledge",
    "jemhopqa": "GLP_general_knowledge",
    "niilc": "GLP_general_knowledge",
    "aio": "GLP_general_knowledge",
    "stem": "GLP_general_knowledge",
    "jmmlu": "GLP_expert_knowledge",
    "mmlu_prox_ja": "GLP_expert_knowledge",
    "hle": "GLP_expert_knowledge",
    
    # 基礎的言語性能
    "jnli": "GLP_semantic_analysis",
    "janli": "GLP_semantic_analysis",
    "jsem": "GLP_semantic_analysis",
    "jsick": "GLP_semantic_analysis",
    "jamp": "GLP_semantic_analysis",
    "jcola-in-domain": "GLP_syntactic_analysis",
    "jcola-out-of-domain": "GLP_syntactic_analysis",
    "jblimp": "GLP_syntactic_analysis",
    
    # アプリケーション開発
    "coding": "GLP_coding",
    "jhumaneval": "GLP_coding",
    "swebench": "GLP_coding",
    "bfcl": "GLP_function_calling",
    
    # アラインメント
    "commonsensemoralja": "ALT_ethics_moral",
    "toxicity": "ALT_toxicity",
    "jbbq": "ALT_bias",
    "jtruthfulqa": "ALT_truthfulness",
    "hallulens": "ALT_truthfulness",
    
    # その他（旧カテゴリ、互換性のため残す）
    "extraction": "GLP_entity_extraction",
}


