# ruff: noqa

import json
from pathlib import Path
from collections import defaultdict, OrderedDict
import numpy as np

RESULTS_DIR = "../scripts/results/results"

# Define MTEB task type mappings
TASK_TYPE_MAPPING = {
    # Retrieval tasks
    "BelebeleRetrieval": "Rtrvl",
    "SlovakSumRetrieval": "Rtrvl",
    "SMESumRetrieval": "Rtrvl",
    "SKQuadRetrieval": "Rtrvl",
    "WebFAQRetrieval": "Rtrvl",
    # Classification tasks
    "SlovakHateSpeechClassification.v2": "Clf",
    "SlovakMovieReviewSentimentClassification.v2": "Clf",
    "SIB200Classification": "Clf",
    "MultilingualSentimentClassification": "Clf",
    "SlovakParlaSentClassification": "Clf",
    "MultiEupSlovakPartyClassification": "Clf",
    "MultiEupSlovakGenderClassification": "Clf",
    # MultilabelClassification tasks
    # 'MultiEURLEXMultilabelClassification': 'MClf',
    # Clustering tasks
    "SIB200ClusteringS2S": "Clust",
    "PravdaSKTagClustering": "Clust",
    "PravdaSKURLClustering": "Clust",
    "SlovakSumURLClustering": "Clust",
    "SMESumCategoryClustering": "Clust",
    # Reranking tasks
    "SkQuadReranking": "Rrnk",
    "SlovakPharmacyDrMaxReranking": "Rrnk",
    "SlovakPharmacyMojaLekarenReranking": "Rrnk",
    # STS tasks
    "SlovakSTS": "STS",
    "SlovakSumSTS": "STS",
    # Pair Classification
    "SlovakNLI": "PrClf",
    "SlovakRTE": "PrClf",
    "DemagogSKNLI": "PrClf",
    # BitextMining
    "OpusSlovakEnglishBitextMining": "Btxt",
    "Tatoeba": "Btxt",
    "FloresBitextMining": "Btxt",
    "NTREXBitextMining": "Btxt",
    "WebFAQBitextMiningQuestions": "Btxt",
    "WebFAQBitextMiningQAs": "Btxt",
}

SPLITS = set()


def extract_main_score(task_data, model_name: str = None):
    """Extract the main score from a task result"""
    scores = task_data.get("scores", {})

    for split in scores:
        SPLITS.add(split)

    relevant_splits = set.intersection(set(scores), ["test", "devtest", "default"])

    if not relevant_splits:
        print(
            f"Err: Missing split scores data for task: {task_data['task_name']} ({model_name})"
        )
    else:
        for split in relevant_splits:
            split_scores = scores[split]
            if isinstance(split_scores, list):
                main_scores = [
                    entry.get("main_score", np.nan) for entry in split_scores
                ]
                return np.nanmean(main_scores) if main_scores else np.nan
            elif isinstance(split_scores, dict):
                return split_scores.get("main_score", np.nan)

    return np.nan


def load_results_from_directory(results_dir=RESULTS_DIR, task_types: list[str] = None):
    """Load evaluation results from the results directory.

    Args:
        results_dir: Directory containing the results
        task_types: Optional list of task types to extract (e.g. ['Clf', 'STS']).
                   If None, extract all known task types.
    """
    results_dir = Path(results_dir)
    model_results = defaultdict(lambda: defaultdict(dict))

    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name

        for revision_dir in model_dir.iterdir():
            if not revision_dir.is_dir():
                continue

            for json_file in revision_dir.glob("*.json"):
                if json_file.name == "model_meta.json":
                    continue

                task_name = json_file.stem

                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        task_data = json.load(f)

                    main_score = extract_main_score(task_data, model_name=model_name)
                    task_type = TASK_TYPE_MAPPING.get(task_name, "Unknown")

                    # Only include tasks of requested types (if specified) and with valid scores
                    if not np.isnan(main_score) and task_type != "Unknown":
                        if task_types is None or task_type in task_types:
                            model_results[model_name][task_type][task_name] = main_score

                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error reading {json_file}: {e}")

    return model_results


def format_model_name(model_name):
    """Format model name for LaTeX (escape underscores)"""
    # Replace double underscores with a formatted version
    return model_name.replace("__", "/").replace("_", "\\_").split("/")[-1]


def latex_escape(text: str) -> str:
    """Escape underscores for LaTeX in arbitrary text (e.g., task names)."""
    return text.replace("_", "\\_")


def format_param_count(n_params):
    """Format parameter count for display (e.g., 85M, 1.3B)."""
    if n_params is None or (isinstance(n_params, float) and np.isnan(n_params)):
        return "-"
    try:
        n_params = float(n_params)
    except (TypeError, ValueError):
        return "-"
    if n_params >= 1_000_000_000:
        val = n_params / 1_000_000_000
        txt = f"{val:.1f}".rstrip("0").rstrip(".")
        return f"{txt}B"
    if n_params >= 1_000_000:
        return f"{n_params / 1_000_000:.0f}M"
    return f"{n_params:.0f}"


def format_embed_dim(embed_dim):
    """Format embedding dimension for display."""
    if embed_dim is None or (isinstance(embed_dim, float) and np.isnan(embed_dim)):
        return "-"
    try:
        return f"{int(embed_dim)}"
    except (TypeError, ValueError):
        return "-"


def create_latex_table(
    model_results,
    caption=None,
    label=None,
    model_sizes=None,
    embed_dims=None,
    section_breaks=None,
):
    """Create a LaTeX table with models as rows and task type averages as columns, plus overall averages by type and by task.

    Parameters:
    - caption: Optional custom caption for the LaTeX table.
    - label: Optional custom label for the LaTeX table.
    """
    if caption is None:
        caption = "skMTEB Results Summary - Average Scores by Task Type (\\%)"
    if label is None:
        label = "tab:mteb_results"

    # Prepare data
    table_data = []

    for model_name, task_types in model_results.items():
        if task_types is None and model_name.startswith("__SECTION__"):
            title = model_name.split(":", 1)[1] if ":" in model_name else model_name
            table_data.append({"__section_title__": title})
            continue
        row = {"Model": format_model_name(model_name)}
        if model_sizes is not None:
            row["Params"] = format_param_count(model_sizes.get(model_name))
        if embed_dims is not None:
            row["EmbedDim"] = format_embed_dim(embed_dims.get(model_name))

        # Calculate an average for each task type
        for task_type, task_scores in task_types.items():
            avg_score = (
                np.mean(list(task_scores.values())) * 100
            )  # Convert to percentage
            row[task_type] = avg_score

        # Calculate overall averages
        all_scores = [
            score
            for task_scores in task_types.values()
            for score in task_scores.values()
        ]
        if all_scores:
            # Average over tasks (each task equally weighted)
            row["AvgTasks"] = np.mean(all_scores) * 100

        # Average over task types (each type equally weighted)
        type_avg_values = [row[tt] for tt in task_types.keys() if tt in row]
        if type_avg_values:
            row["AvgTypes"] = np.mean(type_avg_values)

        table_data.append(row)

    # Sort by average score (descending) unless using sectioned ordering
    has_section_headers = any("__section_title__" in r for r in table_data)
    if not section_breaks and not has_section_headers:
        table_data.sort(key=lambda x: x.get("AvgTasks", 0), reverse=True)

    # Get all task types (columns)
    all_task_types = sorted(
        set(
            task_type
            for row in table_data
            for task_type in row.keys()
            if task_type not in ["Model", "Params", "EmbedDim", "AvgTasks", "AvgTypes"]
            and task_type != "__section_title__"
        )
    )

    # Build LaTeX table
    latex = list()
    latex.append("\\begin{table*}[t]")
    latex.append("\\centering")
    latex.append("\\small")

    # Column specification
    num_task_types = len(all_task_types)
    has_params = model_sizes is not None
    has_embed_dim = embed_dims is not None
    num_cols = (
        3 + num_task_types + (1 if has_params else 0) + (1 if has_embed_dim else 0)
    )  # Model + Params? + EmbedDim? + AvgTasks + AvgTypes + task types
    col_spec = "l" + "r" * (num_cols - 1)
    latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex.append("\\toprule")

    # Header
    if has_params or has_embed_dim:
        meta_cols = (1 if has_params else 0) + (1 if has_embed_dim else 0)
        header_0 = f"& {'& ' * (meta_cols - 1)}\\multicolumn{{2}}{{c}}{{\\textbf{{Average Across}}}} & \\multicolumn{{{num_task_types}}}{{c}}{{\\textbf{{Average per Task Type}}}}"
        header_1 = ["\\textbf{Model} (\\(\\downarrow\\))"]
        if has_params:
            header_1.append("\\textbf{Params}")
        if has_embed_dim:
            header_1.append("\\textbf{Dim}")
        header_1 += ["\\textbf{All}", "\\textbf{Type}"] + [
            f"\\textbf{{{tt}}}" for tt in all_task_types
        ]
    else:
        header_0 = f"& \\multicolumn{{2}}{{c}}{{\\textbf{{Average Across}}}} & \\multicolumn{{{num_task_types}}}{{c}}{{\\textbf{{Average per Task Type}}}}"
        header_1 = [
            "\\textbf{Model} (\\(\\downarrow\\))",
            "\\textbf{All}",
            "\\textbf{Type}",
        ] + [f"\\textbf{{{tt}}}" for tt in all_task_types]
    latex.append(header_0 + " \\\\")
    if has_params or has_embed_dim:
        left_start = 2 + (1 if has_params else 0) + (1 if has_embed_dim else 0)
        left_end = left_start + 1
        right_start = left_end + 1
    else:
        left_start = 2
        left_end = 3
        right_start = 4
    right_end = right_start + num_task_types - 1
    latex.append(
        f"\\cmidrule(lr){{{left_start}-{left_end}}} \\cmidrule(lr){{{right_start}-{right_end}}}"
        + " \\\\"
    )
    latex.append(" & ".join(header_1) + " \\\\")
    latex.append("\\midrule \n")

    # Number of datasets per category
    task_types_size = [
        len([x for x in TASK_TYPE_MAPPING.values() if x == task_type])
        for task_type in all_task_types
    ]
    format_size = lambda x: f"\\textcolor{{gray!75}}{{({x})}}"
    type_count = len(all_task_types)
    datasets_cells = [
        "\\textcolor{gray!75}{Number of datasets (\\(\\rightarrow\\))}",
    ]
    if has_params:
        datasets_cells.append("\\textcolor{gray!75}{-}")
    if has_embed_dim:
        datasets_cells.append("\\textcolor{gray!75}{-}")
    datasets_cells += [
        format_size(sum(task_types_size)),
        format_size(type_count),
        *list(map(format_size, task_types_size)),
    ]
    datasets_line = " & ".join(datasets_cells) + " \\\\"
    latex.append(datasets_line)
    latex.append("\\midrule \n")

    # Determine the best and second-best score per category (for bolding/underlining)
    best_per_type = {}
    second_per_type = {}
    for tt in all_task_types:
        col_values = [
            r.get(tt, np.nan)
            for r in table_data
            if "__section_title__" not in r
        ]
        col_values = [v for v in col_values if not np.isnan(v)]
        if col_values:
            unique_sorted = sorted(set(col_values), reverse=True)
            best_per_type[tt] = unique_sorted[0]
            second_per_type[tt] = unique_sorted[1] if len(unique_sorted) > 1 else np.nan
        else:
            best_per_type[tt] = np.nan
            second_per_type[tt] = np.nan
    best_avg_tasks = max(
        [
            r.get("AvgTasks", np.nan)
            for r in table_data
            if "__section_title__" not in r
            and not np.isnan(r.get("AvgTasks", np.nan))
        ],
        default=np.nan,
    )
    second_avg_tasks = np.nan
    avg_tasks_vals = [
        r.get("AvgTasks", np.nan) for r in table_data if "__section_title__" not in r
    ]
    avg_tasks_vals = [v for v in avg_tasks_vals if not np.isnan(v)]
    if avg_tasks_vals:
        uniq = sorted(set(avg_tasks_vals), reverse=True)
        second_avg_tasks = uniq[1] if len(uniq) > 1 else np.nan
    best_avg_types = max(
        [
            r.get("AvgTypes", np.nan)
            for r in table_data
            if "__section_title__" not in r
            and not np.isnan(r.get("AvgTypes", np.nan))
        ],
        default=np.nan,
    )
    second_avg_types = np.nan
    avg_types_vals = [
        r.get("AvgTypes", np.nan) for r in table_data if "__section_title__" not in r
    ]
    avg_types_vals = [v for v in avg_types_vals if not np.isnan(v)]
    if avg_types_vals:
        uniq = sorted(set(avg_types_vals), reverse=True)
        second_avg_types = uniq[1] if len(uniq) > 1 else np.nan

    # Data rows
    bolded_done = {tt: False for tt in all_task_types}
    underlined_done = {tt: False for tt in all_task_types}
    bolded_avg_tasks_done = False
    bolded_avg_types_done = False
    underlined_avg_tasks_done = False
    underlined_avg_types_done = False
    section_breaks = set(section_breaks or [])
    for i, row in enumerate(table_data):
        if "__section_title__" in row:
            title = row["__section_title__"]
            if not latex or latex[-1] != "\\midrule":
                latex.append("\\midrule")
            latex.append(
                f"\\multicolumn{{{num_cols}}}{{l}}{{\\textit{{{title}}}}} \\\\"
            )
            continue
        model_name = row["Model"]
        avg_tasks = row.get("AvgTasks", np.nan)
        avg_types = row.get("AvgTypes", np.nan)

        # Build row
        row_str = f"{model_name} & "
        if has_params:
            row_str += f"{row.get('Params', '-')}" + " & "
        if has_embed_dim:
            row_str += f"{row.get('EmbedDim', '-')}" + " & "

        if not np.isnan(avg_tasks):
            formatted_avg_tasks = f"{avg_tasks:.2f}"
            is_best_avg_tasks = (not np.isnan(best_avg_tasks)) and (
                abs(avg_tasks - best_avg_tasks) < 1e-9
            )
            is_second_avg_tasks = (not np.isnan(second_avg_tasks)) and (
                abs(avg_tasks - second_avg_tasks) < 1e-9
            )
            if is_best_avg_tasks and not bolded_avg_tasks_done:
                row_str += f"\\textbf{{{formatted_avg_tasks}}}"
                bolded_avg_tasks_done = True
            elif is_second_avg_tasks and not underlined_avg_tasks_done:
                row_str += f"\\underline{{{formatted_avg_tasks}}}"
                underlined_avg_tasks_done = True
            else:
                row_str += formatted_avg_tasks
        else:
            row_str += "-"
        row_str += " & "
        if not np.isnan(avg_types):
            formatted_avg_types = f"{avg_types:.2f}"
            is_best_avg_types = (not np.isnan(best_avg_types)) and (
                abs(avg_types - best_avg_types) < 1e-9
            )
            is_second_avg_types = (not np.isnan(second_avg_types)) and (
                abs(avg_types - second_avg_types) < 1e-9
            )
            if is_best_avg_types and not bolded_avg_types_done:
                row_str += f"\\textbf{{{formatted_avg_types}}}"
                bolded_avg_types_done = True
            elif is_second_avg_types and not underlined_avg_types_done:
                row_str += f"\\underline{{{formatted_avg_types}}}"
                underlined_avg_types_done = True
            else:
                row_str += formatted_avg_types
        else:
            row_str += "-"

        for task_type in all_task_types:
            score = row.get(task_type, np.nan)
            if not np.isnan(score):
                formatted = f"{score:.2f}"
                is_best = (not np.isnan(best_per_type[task_type])) and (
                    abs(score - best_per_type[task_type]) < 1e-9
                )
                is_second = (not np.isnan(second_per_type[task_type])) and (
                    abs(score - second_per_type[task_type]) < 1e-9
                )
                if is_best and not bolded_done[task_type]:
                    row_str += f" & \\textbf{{{formatted}}}"
                    bolded_done[task_type] = True
                elif is_second and not underlined_done[task_type]:
                    row_str += f" & \\underline{{{formatted}}}"
                    underlined_done[task_type] = True
                else:
                    row_str += f" & {formatted}"
            else:
                row_str += " & -"

        row_str += " \\\\"
        latex.append(row_str)
        if i in section_breaks:
            latex.append("\\midrule")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append(f"\\caption{{{caption}}}")
    latex.append(f"\\label{{{label}}}")
    latex.append("\\end{table*}")

    return "\n".join(latex)


def create_classification_table(
    model_task_results,
    caption=None,
    label=None,
    model_sizes=None,
    embed_dims=None,
    section_breaks=None,
):
    """Create a LaTeX table for Classification tasks only with per-task columns and overall average.
    model_task_results: dict[model][task] = score (0-1)
    """
    if caption is None:
        caption = "skMTEB Classification Results - Per-Task Scores (\\%)"
    if label is None:
        label = "tab:mteb_results_clf"

    # Collect all classification tasks present in results
    task_type = "Clf"
    all_tasks = sorted(
        {
            task
            for tasks in model_task_results.values()
            if tasks is not None
            for task in tasks[task_type].keys()
        }
    )

    # Compute best and second-best per task for bolding/underlining
    best_per_task = {}
    second_per_task = {}
    for t in all_tasks:
        vals = [
            model_task_results[m][task_type].get(t, np.nan)
            for m in model_task_results
            if model_task_results[m] is not None
        ]
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            unique_sorted = sorted(set(vals), reverse=True)
            best_per_task[t] = unique_sorted[0] * 100
            second_per_task[t] = (
                unique_sorted[1] * 100 if len(unique_sorted) > 1 else np.nan
            )
        else:
            best_per_task[t] = np.nan
            second_per_task[t] = np.nan

    # Best/second-best for Avg column
    avg_vals = []
    for model_name, task_types in model_task_results.items():
        tasks = task_types.get(task_type) if task_types else None
        if not tasks:
            continue
        available = [
            tasks[t] for t in all_tasks if t in tasks and not np.isnan(tasks[t])
        ]
        if available:
            avg_vals.append(float(np.mean(available) * 100))
    avg_vals = [v for v in avg_vals if not np.isnan(v)]
    best_avg = max(avg_vals) if avg_vals else np.nan
    second_avg = np.nan
    if avg_vals:
        uniq = sorted(set(avg_vals), reverse=True)
        second_avg = uniq[1] if len(uniq) > 1 else np.nan

    # Build table rows
    rows = []
    for model_name, task_types in model_task_results.items():
        if task_types is None and model_name.startswith("__SECTION__"):
            rows.append({"__section_title__": model_name.split(":", 1)[1]})
            continue
        tasks = task_types.get(task_type) if task_types else None
        if not tasks:
            continue
        row = {"Model": format_model_name(model_name)}
        if model_sizes is not None:
            row["Params"] = format_param_count(model_sizes.get(model_name))
        if embed_dims is not None:
            row["EmbedDim"] = format_embed_dim(embed_dims.get(model_name))
        # Per task scores (percent)
        for t in all_tasks:
            v = tasks.get(t, np.nan)
            row[t] = (v * 100) if not np.isnan(v) else np.nan
        # Average over available tasks for this model
        available = [
            tasks[t] for t in all_tasks if t in tasks and not np.isnan(tasks[t])
        ]
        row["AvgTasks"] = (np.mean(available) * 100) if available else np.nan
        rows.append(row)

    # Sort by average descending unless using sectioned ordering
    has_section_headers = any("__section_title__" in r for r in rows)
    if not section_breaks and not has_section_headers:
        rows.sort(key=lambda x: x.get("AvgTasks", 0), reverse=True)

    # LaTeX build
    latex = []
    latex.append("\\begin{table*}[t]")
    latex.append("\\centering")
    latex.append("\\small")

    has_params = model_sizes is not None
    has_embed_dim = embed_dims is not None
    num_cols = (
        2 + len(all_tasks) + (1 if has_params else 0) + (1 if has_embed_dim else 0)
    )  # Model + Params? + Dim? + tasks + Avg
    col_spec = "l" + "r" * (num_cols - 1)
    latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex.append("\\toprule")

    headers = ["\\textbf{Model} (\\(\\downarrow\\))"]
    if has_params:
        headers.append("\\textbf{Params}")
    if has_embed_dim:
        headers.append("\\textbf{Dim}")
    headers += [
        f"\\rotatebox{{90}}{{\\textbf{{{latex_escape(t)}}}}}" for t in all_tasks
    ] + ["\\textbf{Avg}"]
    latex.append(" & ".join(headers) + " \\\\")
    latex.append("\\midrule \n")

    # Bold tracking per task
    bolded_done = {t: False for t in all_tasks}
    underlined_done = {t: False for t in all_tasks}
    bolded_avg_done = False
    underlined_avg_done = False

    section_breaks = set(section_breaks or [])
    for i, row in enumerate(rows):
        if "__section_title__" in row:
            if not latex or latex[-1] != "\\midrule":
                latex.append("\\midrule")
            latex.append(
                f"\\multicolumn{{{num_cols}}}{{l}}{{\\textit{{{row['__section_title__']}}}}} \\\\"
            )
            continue
        parts = [row["Model"]]
        if has_params:
            parts.append(row.get("Params", "-"))
        if has_embed_dim:
            parts.append(row.get("EmbedDim", "-"))
        for t in all_tasks:
            val = row.get(t, np.nan)
            if np.isnan(val):
                parts.append("-")
            else:
                formatted = f"{val:.2f}"
                is_best = (not np.isnan(best_per_task[t])) and (
                    abs(val - best_per_task[t]) < 1e-9
                )
                is_second = (not np.isnan(second_per_task[t])) and (
                    abs(val - second_per_task[t]) < 1e-9
                )
                if is_best and not bolded_done[t]:
                    parts.append(f"\\textbf{{{formatted}}}")
                    bolded_done[t] = True
                elif is_second and not underlined_done[t]:
                    parts.append(f"\\underline{{{formatted}}}")
                    underlined_done[t] = True
                else:
                    parts.append(formatted)
        avg_val = row.get("AvgTasks", np.nan)
        if np.isnan(avg_val):
            parts.append("-")
        else:
            formatted_avg = f"{avg_val:.2f}"
            is_best_avg = (not np.isnan(best_avg)) and (
                abs(avg_val - best_avg) < 1e-9
            )
            is_second_avg = (not np.isnan(second_avg)) and (
                abs(avg_val - second_avg) < 1e-9
            )
            if is_best_avg and not bolded_avg_done:
                parts.append(f"\\textbf{{{formatted_avg}}}")
                bolded_avg_done = True
            elif is_second_avg and not underlined_avg_done:
                parts.append(f"\\underline{{{formatted_avg}}}")
                underlined_avg_done = True
            else:
                parts.append(formatted_avg)
        latex.append(" & ".join(parts) + " \\\\")
        if i in section_breaks:
            latex.append("\\midrule")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append(f"\\caption{{{caption}}}")
    latex.append(f"\\label{{{label}}}")
    latex.append("\\end{table*}")

    return "\n".join(latex)


def create_all_tasks_table(
    model_task_results,
    caption=None,
    label=None,
    model_sizes=None,
    embed_dims=None,
    section_breaks=None,
):
    """Create a LaTeX table for all tasks (across all types) with per-task columns and overall average.
    model_task_results: dict[model][task_type][task] = score (0-1)
    """
    if caption is None:
        caption = "skMTEB Results - Per-Task Scores Across All Tasks (\\%)"
    if label is None:
        label = "tab:mteb_results_all_tasks"

    # Collect all tasks and their types
    task_to_type = {}
    for model_name, type_dict in model_task_results.items():
        if not type_dict:
            continue
        for ttype, tasks in type_dict.items():
            for tname in tasks.keys():
                # Prefer known mapping but fall back to the observed type
                task_to_type[tname] = TASK_TYPE_MAPPING.get(tname, ttype)

    # Stable ordering: by type then by task name
    all_tasks = sorted(task_to_type.keys(), key=lambda t: (task_to_type.get(t, ""), t))

    # Compute best and second-best per task for bolding/underlining (percent)
    best_per_task = {}
    second_per_task = {}
    for t in all_tasks:
        ttype = task_to_type[t]
        vals = []
        for m in model_task_results:
            if model_task_results[m] is None:
                continue
            v = model_task_results[m].get(ttype, {}).get(t, np.nan)
            if not np.isnan(v):
                vals.append(v)
        if vals:
            unique_sorted = sorted(set(vals), reverse=True)
            best_per_task[t] = unique_sorted[0] * 100
            second_per_task[t] = (
                unique_sorted[1] * 100 if len(unique_sorted) > 1 else np.nan
            )
        else:
            best_per_task[t] = np.nan
            second_per_task[t] = np.nan

    # Best/second-best for Avg column
    avg_vals = []
    for model_name, type_dict in model_task_results.items():
        if not type_dict:
            continue
        available_vals = []
        for t in all_tasks:
            ttype = task_to_type[t]
            v = type_dict.get(ttype, {}).get(t, np.nan)
            if not np.isnan(v):
                available_vals.append(v)
        if available_vals:
            avg_vals.append(float(np.mean(available_vals) * 100))
    avg_vals = [v for v in avg_vals if not np.isnan(v)]
    best_avg = max(avg_vals) if avg_vals else np.nan
    second_avg = np.nan
    if avg_vals:
        uniq = sorted(set(avg_vals), reverse=True)
        second_avg = uniq[1] if len(uniq) > 1 else np.nan

    # Build table rows
    rows = []
    for model_name, type_dict in model_task_results.items():
        if not type_dict:
            continue
        row = {"Model": format_model_name(model_name)}
        if model_sizes is not None:
            row["Params"] = format_param_count(model_sizes.get(model_name))
        if embed_dims is not None:
            row["EmbedDim"] = format_embed_dim(embed_dims.get(model_name))
        available_vals = []
        for t in all_tasks:
            ttype = task_to_type[t]
            v = type_dict.get(ttype, {}).get(t, np.nan)
            if not np.isnan(v):
                row[t] = v * 100
                available_vals.append(v)
            else:
                row[t] = np.nan
        row["AvgTasks"] = (np.mean(available_vals) * 100) if available_vals else np.nan
        rows.append(row)

    # Sort by average descending unless using sectioned ordering
    has_section_headers = any("__section_title__" in r for r in rows)
    if not section_breaks and not has_section_headers:
        rows.sort(key=lambda x: x.get("AvgTasks", 0), reverse=True)

    # LaTeX build
    latex = []
    latex.append("\\begin{table*}[t]")
    latex.append("\\centering")
    latex.append("\\small")

    has_params = model_sizes is not None
    has_embed_dim = embed_dims is not None
    num_cols = (
        2 + len(all_tasks) + (1 if has_params else 0) + (1 if has_embed_dim else 0)
    )  # Model + Params? + Dim? + tasks + Avg
    col_spec = "l" + "r" * (num_cols - 1)
    latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex.append("\\toprule")

    # Headers with type prefix for clarity
    headers = ["\\textbf{Model} (\\(\\downarrow\\))"]
    if has_params:
        headers.append("\\textbf{Params}")
    if has_embed_dim:
        headers.append("\\textbf{Dim}")
    for t in all_tasks:
        ttype = task_to_type.get(t, "")
        header_txt = f"{ttype}: {latex_escape(t)}" if ttype else latex_escape(t)
        headers.append(f"\\rotatebox{{90}}{{\\textbf{{{header_txt}}}}}")
    headers.append("\\textbf{Avg}")
    latex.append(" & ".join(headers) + " \\\\")
    latex.append("\\midrule \n")

    # Bold tracking per task
    bolded_done = {t: False for t in all_tasks}
    underlined_done = {t: False for t in all_tasks}
    bolded_avg_done = False
    underlined_avg_done = False

    section_breaks = set(section_breaks or [])
    for i, row in enumerate(rows):
        if "__section_title__" in row:
            if not latex or latex[-1] != "\\midrule":
                latex.append("\\midrule")
            latex.append(
                f"\\multicolumn{{{num_cols}}}{{l}}{{\\textit{{{row['__section_title__']}}}}} \\\\"
            )
            continue
        parts = [row["Model"]]
        if has_params:
            parts.append(row.get("Params", "-"))
        if has_embed_dim:
            parts.append(row.get("EmbedDim", "-"))
        for t in all_tasks:
            val = row.get(t, np.nan)
            if np.isnan(val):
                parts.append("-")
            else:
                formatted = f"{val:.2f}"
                is_best = (not np.isnan(best_per_task[t])) and (
                    abs(val - best_per_task[t]) < 1e-9
                )
                is_second = (not np.isnan(second_per_task[t])) and (
                    abs(val - second_per_task[t]) < 1e-9
                )
                if is_best and not bolded_done[t]:
                    parts.append(f"\\textbf{{{formatted}}}")
                    bolded_done[t] = True
                elif is_second and not underlined_done[t]:
                    parts.append(f"\\underline{{{formatted}}}")
                    underlined_done[t] = True
                else:
                    parts.append(formatted)
        avg_val = row.get("AvgTasks", np.nan)
        if np.isnan(avg_val):
            parts.append("-")
        else:
            formatted_avg = f"{avg_val:.2f}"
            is_best_avg = (not np.isnan(best_avg)) and (
                abs(avg_val - best_avg) < 1e-9
            )
            is_second_avg = (not np.isnan(second_avg)) and (
                abs(avg_val - second_avg) < 1e-9
            )
            if is_best_avg and not bolded_avg_done:
                parts.append(f"\\textbf{{{formatted_avg}}}")
                bolded_avg_done = True
            elif is_second_avg and not underlined_avg_done:
                parts.append(f"\\underline{{{formatted_avg}}}")
                underlined_avg_done = True
            else:
                parts.append(formatted_avg)
        latex.append(" & ".join(parts) + " \\\\")
        if i in section_breaks:
            latex.append("\\midrule")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append(f"\\caption{{{caption}}}")
    latex.append(f"\\label{{{label}}}")
    latex.append("\\end{table*}")

    return "\n".join(latex)


def load_model_metadata(metadata_type: str, results_dir=RESULTS_DIR):
    """Load requested metadata field from model_meta.json files.
    Args:
        metadata_type: Key to extract from model_meta.json (e.g. 'n_parameters', 'max_tokens')
        results_dir: Directory containing model results
    Returns: dict[model_name] = metadata_value
    """
    results_dir = Path(results_dir)
    model_metadata = {}

    metadata_overrides = {
        "n_parameters": {
            "kinit__slovakbert-sts-stsb": 124_644_864,
            "TUKE-DeutscheTelekom__slovakbert-skquad-mnlr": 124_644_864,
        }
        ,
        "embed_dim": {
            "kinit__slovakbert-sts-stsb": 768,
            "TUKE-DeutscheTelekom__slovakbert-skquad-mnlr": 768,
            "cohere__embed-v4.0": 1536,
        },
    }

    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue

        # Find the first revision dir that contains model_meta.json
        metadata_value = None
        for revision_dir in model_dir.iterdir():
            if not revision_dir.is_dir():
                continue
            meta_file = revision_dir / "model_meta.json"
            if meta_file.exists():
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    value = meta.get(metadata_type)
                    if isinstance(value, (int, float, str, bool)):
                        metadata_value = value
                        break
                    else:
                        print(f"Warn: Invalid {metadata_type} value in {model_dir}")
                except (json.JSONDecodeError, OSError) as e:
                    print(f"Error reading {meta_file}: {e}")
        if metadata_value is not None:
            model_metadata[model_dir.name] = metadata_value

        overrides_for_type = metadata_overrides.get(metadata_type, {})
        if model_dir.name in overrides_for_type:
            model_metadata[model_dir.name] = overrides_for_type[model_dir.name]

    return model_metadata


def split_model_results_by_size(model_results, model_sizes):
    """Split model results into size buckets: small (<130M), base (<350M), large (>=350M)."""
    buckets = {"small": {}, "base": {}, "large": {}}
    missing = []
    for model_name, task_types in model_results.items():
        size = model_sizes.get(model_name)
        if size is None or (isinstance(size, float) and np.isnan(size)):
            missing.append(model_name)
            bucket = "base"
        elif size < 130_000_000:
            bucket = "small"
        elif size < 350_000_000:
            bucket = "base"
        else:
            bucket = "large"
        buckets[bucket][model_name] = task_types
    return buckets, missing


def sort_model_results_by_size(model_results, model_sizes, embed_dims=None):
    """Return model_results ordered by ascending parameter count (missing sizes by embed dim, then name)."""

    def sort_key(item):
        name, _ = item
        size = model_sizes.get(name)
        if size is None or (isinstance(size, float) and np.isnan(size)):
            dim = None if embed_dims is None else embed_dims.get(name)
            if dim is None or (isinstance(dim, float) and np.isnan(dim)):
                return (1, float("inf"), name)
            return (1, float(dim), name)
        return (0, float(size), name)

    return OrderedDict(sorted(model_results.items(), key=sort_key))


def create_size_grouped_tables(
    model_results,
    model_sizes,
    embed_dims,
    table_fn,
    caption_base,
    label_base,
    sort_by_size=False,
):
    """Create a single LaTeX table with size buckets separated by titled sections."""
    api_prefix = "text-embedding"
    api_exact = {"embed-4.0", "embed-v4.0"}
    api_models = {
        name
        for name in model_results.keys()
        if name.split("__")[-1].lower().startswith(api_prefix)
        or name.split("__")[-1] in api_exact
    }

    buckets, missing = split_model_results_by_size(model_results, model_sizes)
    if missing:
        print(
            f"Warn: {len(missing)} models missing n_parameters; defaulting them to 'base' bucket."
        )

    sections = [
        ("small", "Small models (<130M)"),
        ("base", "Base models (>=130M, <350M)"),
        ("large", "Large models (>=350M)"),
    ]

    ordered = OrderedDict()
    for key, _title in sections:
        bucket_results = {
            k: v for k, v in buckets[key].items() if k not in api_models
        }
        if not bucket_results:
            continue
        ordered[f"__SECTION__:{_title}"] = None
        if sort_by_size:
            bucket_results = sort_model_results_by_size(
                bucket_results, model_sizes, embed_dims
            )
        for model_name, task_types in bucket_results.items():
            ordered[model_name] = task_types

    # Append API access models as a final block
    api_results = {k: v for k, v in model_results.items() if k in api_models}
    if api_results:
        ordered[f"__SECTION__:API access models"] = None
        if sort_by_size:
            api_results = sort_model_results_by_size(api_results, model_sizes, embed_dims)
        for model_name, task_types in api_results.items():
            ordered[model_name] = task_types

    return table_fn(
        ordered,
        caption=caption_base,
        label=label_base,
        model_sizes=model_sizes,
        embed_dims=embed_dims,
        section_breaks=None,
    )


def create_model_size_scatter(
    model_results,
    results_dir=RESULTS_DIR,
    output_file="model_size_vs_avg.png",
    annotate=False,
    log_x=True,
    annotate_top_n=6,
    annotate_names=None,
):
    """Create a scatter plot (seaborn) of model size (n_parameters) vs average across all tasks (%).
    - x-axis: n_parameters from model_meta.json
    - y-axis: Avg across all tasks for the model (percent)
    - color: max_tokens from model_meta.json (log-scaled colorbar)
    Each point represents a model.
    Parameters:
    - annotate: if True and annotate_names is None, will label all points (can be cluttered)
    - annotate_top_n: if > 0, label the top-N models by Avg across tasks
    - annotate_names: optional list of model base names to label (directory names or formatted names)
    """
    try:
        import seaborn as sns  # local import to avoid hard dependency if not used
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except ImportError:
        print(
            "seaborn (and/or matplotlib) is not installed; skipping scatter plot generation."
        )
        return

    model_params = load_model_metadata("n_parameters", results_dir)
    model_tokens = load_model_metadata("max_tokens", results_dir)

    xs, ys, labels, toks, raw_names = [], [], [], [], []
    for model_name, task_types in model_results.items():
        # Compute average across all tasks for this model
        all_scores = [
            score
            for task_scores in task_types.values()
            for score in task_scores.values()
        ]
        if not all_scores:
            continue
        avg_percent = float(np.mean(all_scores) * 100.0)
        n_params = model_params.get(model_name)
        max_toks = model_tokens.get(model_name)

        if max_toks is None:
            max_toks = max(model_tokens.values())
        if n_params is None:
            print(f"Err: Missing metadata for {model_name} (n_params={n_params})")
            continue
        xs.append(n_params)
        ys.append(avg_percent)
        labels.append(format_model_name(model_name))
        toks.append(max_toks)
        raw_names.append(model_name)

    if not xs:
        print("No data available to plot model size vs average.")
        return

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(14, 5))

    # Colors by max_tokens (log scale)
    cmap = sns.color_palette("Spectral", as_cmap=True)
    vmin = min(500.0, float(min(toks)))
    vmax = max(50000.0, float(max(toks)))
    norm = LogNorm(vmin=vmin, vmax=vmax)

    # Use seaborn for styling but disable legend; we add a colorbar instead
    sns.scatterplot(
        x=xs,
        y=ys,
        hue=toks,
        palette=cmap,
        hue_norm=norm,
        ax=ax,
        s=70,
        alpha=0.95,
        edgecolor="black",
        linewidth=0.5,
        legend=False,
    )

    # Colorbar
    import matplotlib as mpl

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Max Tokens")

    # Human-readable ticks on the Max Tokens colorbar (log scale)
    try:
        from matplotlib.ticker import LogLocator, FuncFormatter

        # Ticks at 1, 2, 5 per decade
        cbar.locator = LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=10)

        def _human_tokens(val, pos=None):
            # Format decade ticks as 1k/10k/100k etc., and inter-decade ticks as 2 and 5
            if val <= 0:
                return ""
            import math

            exp = int(math.floor(math.log10(val)))
            mant = val / (10**exp)
            if abs(mant - 1.0) < 1e-8:
                if exp >= 9:
                    suffix = "B"
                    base = 9
                elif exp >= 6:
                    suffix = "M"
                    base = 6
                elif exp >= 3:
                    suffix = "K"
                    base = 3
                else:
                    suffix = ""
                    base = 0
                num = int(round(val / (10**base)))
                return f"{num}{suffix}" if suffix else f"{num}"
            elif abs(mant - 2.0) < 1e-8 or abs(mant - 5.0) < 1e-8:
                return "2" if abs(mant - 2.0) < 1e-8 else "5"
            return ""

        cbar.formatter = FuncFormatter(_human_tokens)
        cbar.update_ticks()
    except Exception:
        # If anything goes wrong, fall back silently to default formatting
        pass

    # Annotations
    indices_to_annotate = set()
    if annotate_names:
        # Allow matching by raw dir name or formatted label
        name_set = set(annotate_names)
        for i, (raw, lbl) in enumerate(zip(raw_names, labels)):
            if raw in name_set or lbl in name_set:
                indices_to_annotate.add(i)
    elif annotate_top_n and annotate_top_n > 0:
        # annotate top-N by y
        top_idx = np.argsort(-np.array(ys))[:annotate_top_n]
        indices_to_annotate.update(map(int, top_idx))
    elif annotate:
        indices_to_annotate.update(range(len(xs)))

    for i in indices_to_annotate:
        ax.annotate(
            labels[i],
            (xs[i], ys[i]),
            textcoords="offset points",
            xytext=(-10, 6),
            fontsize=8,
            alpha=0.9,
        )

    if log_x:
        ax.set_xscale("log")
        from matplotlib.ticker import LogLocator, FuncFormatter

        # Place ticks at 1, 2, and 5 within each decade
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))

        def _human_fmt(val, pos=None):
            # Format decade ticks as 100M/1B/10B etc., and inter-decade ticks as 2 and 5
            if val <= 0:
                return ""
            import math

            exp = int(math.floor(math.log10(val)))
            mant = val / (10**exp)
            # Only label intended mantissas 1, 2, 5
            if abs(mant - 1.0) < 1e-8:
                # Choose suffix
                if exp >= 9:
                    suffix = "B"
                    base = 9
                elif exp >= 6:
                    suffix = "M"
                    base = 6
                elif exp >= 3:
                    suffix = "K"
                    base = 3
                else:
                    suffix = ""
                    base = 0
                num = int(round(val / (10**base)))
                return f"{num}{suffix}" if suffix else f"{num}"
            elif abs(mant - 2.0) < 1e-8 or abs(mant - 5.0) < 1e-8:
                # Show just 2 or 5 between decades
                if abs(mant - 2.0) < 1e-8:
                    return "2"
                else:
                    return "5"
            return ""

        ax.xaxis.set_major_formatter(FuncFormatter(_human_fmt))

    # Fix the max X-axis value to 10B (10^10 parameters)
    ax.set_xlim(left=100_000_000, right=10_000_000_000)

    ax.set_xlabel("Number of Parameters")
    ax.set_ylabel("Average across all tasks (%)")
    # ax.set_title('Model Size vs. Average Performance')
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(output_file, dpi=220)
    plt.close(fig)
    print(f"\n✓ Scatter plot saved to '{output_file}'")


def main():
    print(f"Loading results from '{RESULTS_DIR}' directory...")
    model_results = load_results_from_directory(RESULTS_DIR)
    print(SPLITS)

    print(f"Found results for {len(model_results)} models")
    model_sizes = load_model_metadata("n_parameters", RESULTS_DIR)
    embed_dims = load_model_metadata("embed_dim", RESULTS_DIR)
    output_dir = Path("../sandbox")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a LaTeX summary table
    print("\nCreating summary table (per task type) ...")
    latex_table = create_size_grouped_tables(
        model_results,
        model_sizes,
        embed_dims,
        create_latex_table,
        caption_base="skMTEB Results Summary - Average Scores by Task Type (\\%)",
        label_base="tab:mteb_results",
        sort_by_size=True,
    )

    output_file = output_dir / "mteb_results_table.tex"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(latex_table)

    print(f"\n✓ LaTeX table saved to '{output_file}'")
    print("\n" + "=" * 80)
    print("LaTeX Table Preview:")
    print("=" * 80)
    print(latex_table)
    print("=" * 80)

    # Create Classification-only LaTeX table (per-task columns)
    print("\nCreating classification-only table (per-task) ...")
    latex_table_clf = create_size_grouped_tables(
        model_results,
        model_sizes,
        embed_dims,
        create_classification_table,
        caption_base="skMTEB Classification Results - Per-Task Scores (\\%)",
        label_base="tab:mteb_results_clf",
        sort_by_size=True,
    )

    output_file_clf = output_dir / "mteb_results_table_clf.tex"
    with open(output_file_clf, "w", encoding="utf-8") as f:
        f.write(latex_table_clf)
    print(f"\n✓ LaTeX classification table saved to '{output_file_clf}'")
    print("\n" + "=" * 80)
    print("Classification LaTeX Table Preview:")
    print("=" * 80)
    print(latex_table_clf)
    print("=" * 80)

    # Create All tasks LaTeX table
    print("\nCreating complete table (per-task) ...")
    latex_table_all = create_size_grouped_tables(
        model_results,
        model_sizes,
        embed_dims,
        create_all_tasks_table,
        caption_base="skMTEB Results - Per-Task Scores Across All Tasks (\\%)",
        label_base="tab:mteb_results_all_tasks",
        sort_by_size=True,
    )

    output_file_all = output_dir / "mteb_results_table_all.tex"
    with open(output_file_all, "w", encoding="utf-8") as f:
        f.write(latex_table_all)
    print(f"\n✓ LaTeX complete table saved to '{output_file_all}'")
    print("\n" + "=" * 80)
    print("Complete LaTeX Table Preview:")
    print("=" * 80)
    print(latex_table_all)
    print("=" * 80)

    # Create model size vs average scatter plot
    print("\nCreating model size vs average scatter plot ...")
    create_model_size_scatter(
        model_results,
        RESULTS_DIR,
        str(output_dir / "model_size_vs_avg.png"),
        annotate=True,
        log_x=True,
        annotate_top_n=30,
        annotate_names=[
            "Qwen3-Embedding-8B",
            "Qwen3-Embedding-4B",
            "Qwen3-Embedding-4B-8k",
            "Qwen3-Embedding-0.6B",
            "multilingual-e5-large-instruct",
            "multilingual-e5-large",
            "multilingual-e5-base",
            "multilingual-e5-small",
            # 'bge-m3',
            # 'nomic-embed-text-v2-moe',
            "nomic-embed-text-v1.5",
            # 'gte-multilingual-base',
            "jina-embeddings-v4",
            # 'jina-embeddings-v3',
            "embeddinggemma-300m",
            # 'paraphrase-multilingual-mpnet-base-v2',
            # 'paraphrase-multilingual-MiniLM-L12-v2',
            "LaBSE",
            # 'granite-embedding-278m-multilingual',
            "granite-embedding-107m-multilingual",
            "slovakbert-skquad-mnlr",
            "slovakbert-sts-stsb",
            # 'static-similarity-mrl-multilingual-v1',
        ],
    )

    # Also print some statistics
    print("\nTask Type Statistics:")
    task_type_counts = defaultdict(int)
    for model_name, task_types in model_results.items():
        for task_type in task_types.keys():
            task_type_counts[task_type] += 1

    for task_type, count in sorted(task_type_counts.items()):
        print(f"  {task_type}: {count} models evaluated")


if __name__ == "__main__":
    main()
