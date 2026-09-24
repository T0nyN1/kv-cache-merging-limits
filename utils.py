import ast
import os
from datetime import datetime
from typing import Dict, Any, Optional

import pandas as pd
import torch


def set_device():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    return device


def export_results(summary_results: Dict[str, Dict[str, Any]], save_dir: Optional[str] = None,
                   filename: Optional[str] = None) -> None:
    def _flatten_result(result: Any, prefix: str = "") -> dict:
        if isinstance(result, str):
            res_str = result.strip()
            if res_str.startswith("{") and res_str.endswith("}"):
                try:
                    result = ast.literal_eval(res_str)
                except (ValueError, SyntaxError):
                    pass

        if not isinstance(result, dict):
            return {prefix if prefix else "Score": result}

        flat = {}
        for k, v in result.items():
            new_key = f"{prefix}_{k}" if prefix else str(k)

            if isinstance(v, str):
                v_str = v.strip()
                if v_str.startswith("{") and v_str.endswith("}"):
                    try:
                        v = ast.literal_eval(v_str)
                    except (ValueError, SyntaxError):
                        pass

            if isinstance(v, dict):
                flat.update(_flatten_result(v, new_key))
            else:
                flat[new_key] = v
        return flat

    formatted_data = {}

    for task_name, method_res in summary_results.items():
        for method_name, raw_result in method_res.items():
            if method_name not in formatted_data:
                formatted_data[method_name] = {}

            flat_dict = _flatten_result(raw_result)

            for key, value in flat_dict.items():
                if key.startswith(f"{task_name}_"):
                    clean_key = key[len(task_name) + 1:]
                elif key.startswith("needlehaystack_"):
                    clean_key = key[len("needlehaystack_"):]
                else:
                    clean_key = key

                formatted_data[method_name][(task_name, clean_key)] = value

    if not formatted_data:
        print("No results found. Export aborted.")
        return

    df = pd.DataFrame.from_dict(formatted_data, orient='index')

    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df = df.reset_index()
    new_columns = [("Task", "Model")] + df.columns.tolist()[1:]
    df.columns = pd.MultiIndex.from_tuples(new_columns)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir if save_dir is not None else "",
                             f"{filename if filename is not None else 'eval_results'}_{timestamp}.csv")
    df.to_csv(save_path, index=False, encoding="utf-8")
    print(f"Results saved to: {save_path}")
