import json
from pathlib import Path
from collections import defaultdict
import numpy as np

# Define MTEB task type mappings
TASK_TYPE_MAPPING = {
    # Retrieval tasks
    'BelebeleRetrieval': 'Rtrvl',
    'SlovakSumRetrieval': 'Rtrvl',
    'SMESumRetrieval': 'Rtrvl',
    'SKQuadRetrieval': 'Rtrvl',
    'WebFAQRetrieval': 'Rtrvl',

    # Classification tasks
    'SlovakHateSpeechClassification.v2': 'Clf',
    'SlovakMovieReviewSentimentClassification.v2': 'Clf',
    'SIB200Classification': 'Clf',
    'MultilingualSentimentClassification': 'Clf',
    'DGurgurovSlovakSentiment': 'Clf',
    'SlovakParlaSentClassification': 'Clf',
    'MultiEupSlovakPartyClassification': 'Clf',
    'MultiEupSlovakGenderClassification': 'Clf',

    # MultilabelClassification tasks
    # 'MultiEURLEXMultilabelClassification': 'MClf',

    # Clustering tasks
    'SIB200ClusteringS2S': 'Clust',
    'PravdaSKTagClustering': 'Clust',
    'PravdaSKURLClustering': 'Clust',
    'SlovakSumURLClustering': 'Clust',
    'SMESumCategoryClustering': 'Clust',

    # Reranking tasks
    'SkQuadReranking': 'Rrnk',
    'SlovakFactCheckReranking': 'Rrnk',

    # STS tasks
    'SlovakSTS': 'STS',

    # Pair Classification
    'SlovakNLI': 'PrClf',
    'SlovakRTE': 'PrClf',
    'DemagogSKNLI': 'PrClf',

    # BitextMining
    'OpusSlovakEnglishBitextMining': 'Btxt',
    'Tatoeba': 'Btxt',
    'FloresBitextMining': 'Btxt',
    'NTREXBitextMining': 'Btxt',
    'WebFAQBitextMiningQuestions': 'Btxt',
    'WebFAQBitextMiningQAs': 'Btxt',
}

SPLITS = set()
def extract_main_score(task_data, model_name: None):
    """Extract the main score from a task result"""
    scores = task_data.get('scores', {})

    for split in scores:
        SPLITS.add(split)

    relevant_splits = set.intersection(set(scores), ['test', 'devtest', 'default'])

    if not relevant_splits:
        print(f"Err: Missing split scores data for task: {task_data['task_name']} ({model_name})")
    else:
        for split in relevant_splits:
            split_scores = scores[split]
            if isinstance(split_scores, list):
                main_scores = [entry.get('main_score', np.nan) for entry in split_scores]
                return np.nanmean(main_scores) if main_scores else np.nan
            elif isinstance(split_scores, dict):
                return split_scores.get('main_score', np.nan)
    
    return np.nan

def load_results_from_directory(results_dir='results'):
    """Load all evaluation results from the results directory"""
    results_dir = Path(results_dir)
    model_results = defaultdict(lambda: defaultdict(list))
    
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        model_name = model_dir.name
        
        for revision_dir in model_dir.iterdir():
            if not revision_dir.is_dir():
                continue
            
            for json_file in revision_dir.glob('*.json'):
                if json_file.name == 'model_meta.json':
                    continue
                
                task_name = json_file.stem
                
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        task_data = json.load(f)
                    
                    main_score = extract_main_score(task_data, model_name=model_name)
                    task_type = TASK_TYPE_MAPPING.get(task_name, 'Unknown')
                    
                    if not np.isnan(main_score) and task_type != 'Unknown':
                        model_results[model_name][task_type].append(main_score)
                        
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error reading {json_file}: {e}")
    
    return model_results

def format_model_name(model_name):
    """Format model name for LaTeX (escape underscores)"""
    # Replace double underscores with a formatted version
    return model_name.replace('__', '/').replace('_', '\\_').split('/')[-1]


def latex_escape(text: str) -> str:
    """Escape underscores for LaTeX in arbitrary text (e.g., task names)."""
    return text.replace('_', '\\_')


def create_latex_table(model_results, caption=None, label=None):
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
        row = {'Model': format_model_name(model_name)}
        
        # Calculate an average for each task type
        for task_type, scores in task_types.items():
            avg_score = np.mean(scores) * 100  # Convert to percentage
            row[task_type] = avg_score
        
        # Calculate overall averages
        all_scores = [score for scores in task_types.values() for score in scores]
        if all_scores:
            # Average over tasks (each task equally weighted)
            row['AvgTasks'] = np.mean(all_scores) * 100
        
        # Average over task types (each type equally weighted)
        type_avg_values = [row[tt] for tt in task_types.keys() if tt in row]
        if type_avg_values:
            row['AvgTypes'] = np.mean(type_avg_values)
        
        table_data.append(row)
    
    # Sort by average score (descending)
    table_data.sort(key=lambda x: x.get('AvgTasks', 0), reverse=True)
    
    # Get all task types (columns)
    all_task_types = sorted(set(
        task_type 
        for row in table_data 
        for task_type in row.keys() 
        if task_type not in ['Model', 'AvgTasks', 'AvgTypes']
    ))
    
    # Build LaTeX table
    latex = list()
    latex.append("\\begin{table*}[p]")
    latex.append("\\centering")
    latex.append("\\small")

    # Column specification
    num_task_types = len(all_task_types)
    num_cols = 3 + num_task_types  # Model + AvgTasks + AvgTypes + task types
    col_spec = "l" + "r" * (num_cols - 1)
    latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex.append("\\toprule")

    # Header
    header_0 = f"& \\multicolumn{{2}}{{c}}{{\\textbf{{Average Across}}}} & \\multicolumn{{{num_task_types}}}{{c}}{{\\textbf{{Average per Task Type}}}}"
    header_1 = ["\\textbf{Model} (\\(\\downarrow\\))", "\\textbf{All}", "\\textbf{Type}"] + [f"\\textbf{{{tt}}}" for tt in all_task_types]
    latex.append(header_0 + " \\\\")
    latex.append("\\cmidrule(lr){2-3} \\cmidrule(lr){4-10}" + " \\\\")
    latex.append(" & ".join(header_1) + " \\\\")
    latex.append("\\midrule \n")

    # Number of datasets per category
    task_types_size = [len([x for x in TASK_TYPE_MAPPING.values() if x == task_type]) for task_type in all_task_types]
    format_size = lambda x: f'\\textcolor{{gray!75}}{{({x})}}'
    type_count = len(all_task_types)
    datasets_cells = [
        "\\textcolor{gray!75}{Number of datasets (\\(\\rightarrow\\))}",
        format_size(sum(task_types_size)),
        format_size(type_count),
        *list(map(format_size, task_types_size)),
    ]
    datasets_line = " & ".join(datasets_cells) + " \\\\" 
    latex.append(datasets_line)
    latex.append("\\midrule \n")

    # Determine the best score per category (for bolding one value per column)
    best_per_type = {}
    for tt in all_task_types:
        col_values = [r.get(tt, np.nan) for r in table_data]
        col_values = [v for v in col_values if not np.isnan(v)]
        best_per_type[tt] = max(col_values) if col_values else np.nan

    # Data rows
    bolded_done = {tt: False for tt in all_task_types}
    for i, row in enumerate(table_data):
        model_name = row['Model']
        avg_tasks = row.get('AvgTasks', np.nan)
        avg_types = row.get('AvgTypes', np.nan)

        # Build row
        row_str = f"{model_name} & "

        row_str += f"{avg_tasks:.2f}" if not np.isnan(avg_tasks) else "-"
        row_str += " & "
        row_str += f"{avg_types:.2f}" if not np.isnan(avg_types) else "-"

        for task_type in all_task_types:
            score = row.get(task_type, np.nan)
            if not np.isnan(score):
                formatted = f"{score:.2f}"
                is_best = (not np.isnan(best_per_type[task_type])) and (abs(score - best_per_type[task_type]) < 1e-9)
                if is_best and not bolded_done[task_type]:
                    row_str += f" & \\textbf{{{formatted}}}"
                    bolded_done[task_type] = True
                else:
                    row_str += f" & {formatted}"
            else:
                row_str += " & -"

        row_str += " \\\\"
        latex.append(row_str)

        # Add a midrule every 5 rows for readability
        if (i + 1) % 5 == 0 and i < len(table_data) - 1:
            latex.append("\\midrule")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append(f"\\caption{{{caption}}}")
    latex.append(f"\\label{{{label}}}")
    latex.append("\\end{table*}")

    return "\n".join(latex)

def load_classification_task_results(results_dir='results'):
    """Load per-task results for classification tasks (Clf) only.
    Returns: dict[model_name][task_name] = main_score (raw, 0-1)
    """
    results_dir = Path(results_dir)
    clf_results = defaultdict(dict)

    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        for revision_dir in model_dir.iterdir():
            if not revision_dir.is_dir():
                continue

            for json_file in revision_dir.glob('*.json'):
                if json_file.name == 'model_meta.json':
                    continue
                task_name = json_file.stem
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        task_data = json.load(f)
                    main_score = extract_main_score(task_data, model_name=model_name)
                    if not np.isnan(main_score) and TASK_TYPE_MAPPING.get(task_name) == 'Clf':
                        clf_results[model_name][task_name] = main_score
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error reading {json_file}: {e}")

    return clf_results


def create_classification_table(model_task_results, caption=None, label=None):
    """Create a LaTeX table for Classification tasks only with per-task columns and overall average.
    model_task_results: dict[model][task] = score (0-1)
    """
    if caption is None:
        caption = "skMTEB Classification Results - Per-Task Scores (\\%)"
    if label is None:
        label = "tab:mteb_results_clf"

    # Collect all classification tasks present in results
    all_tasks = sorted({task for tasks in model_task_results.values() for task in tasks.keys()})

    # Compute best per task for bolding
    best_per_task = {}
    for t in all_tasks:
        vals = [model_task_results[m].get(t, np.nan) for m in model_task_results]
        vals = [v for v in vals if not np.isnan(v)]
        best_per_task[t] = max(vals) * 100 if vals else np.nan

    # Build table rows
    rows = []
    for model_name, tasks in model_task_results.items():
        row = {'Model': format_model_name(model_name)}
        # Per task scores (percent)
        for t in all_tasks:
            v = tasks.get(t, np.nan)
            row[t] = (v * 100) if not np.isnan(v) else np.nan
        # Average over available tasks for this model
        available = [tasks[t] for t in all_tasks if t in tasks and not np.isnan(tasks[t])]
        row['AvgTasks'] = (np.mean(available) * 100) if available else np.nan
        rows.append(row)

    # Sort by average descending
    rows.sort(key=lambda x: x.get('AvgTasks', 0), reverse=True)

    # LaTeX build
    latex = []
    latex.append("\\begin{table*}[p]")
    latex.append("\\centering")
    latex.append("\\small")

    num_cols = 2 + len(all_tasks)  # Model + tasks + Avg
    col_spec = "l" + "r" * (num_cols - 1)
    latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex.append("\\toprule")

    headers = ["\\textbf{Model} (\\(\\downarrow\\))"] + [f"\\rotatebox{{90}}{{\\textbf{{{latex_escape(t)}}}}}" for t in all_tasks] + ["\\textbf{Avg}"]
    latex.append(" & ".join(headers) + " \\\\")
    latex.append("\\midrule \n")

    # Bold tracking per task
    bolded_done = {t: False for t in all_tasks}

    for i, row in enumerate(rows):
        parts = [row['Model']]
        for t in all_tasks:
            val = row.get(t, np.nan)
            if np.isnan(val):
                parts.append("-")
            else:
                formatted = f"{val:.2f}"
                is_best = (not np.isnan(best_per_task[t])) and (abs(val - best_per_task[t]) < 1e-9)
                if is_best and not bolded_done[t]:
                    parts.append(f"\\textbf{{{formatted}}}")
                    bolded_done[t] = True
                else:
                    parts.append(formatted)
        avg_val = row.get('AvgTasks', np.nan)
        parts.append(f"{avg_val:.2f}" if not np.isnan(avg_val) else "-")
        latex.append(" & ".join(parts) + " \\\\")
        if (i + 1) % 5 == 0 and i < len(rows) - 1:
            latex.append("\\midrule")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append(f"\\caption{{{caption}}}")
    latex.append(f"\\label{{{label}}}")
    latex.append("\\end{table*}")

    return "\n".join(latex)



def main():
    print("Loading results from 'results/' directory...")
    model_results = load_results_from_directory('results')
    print(SPLITS)

    print(f"Found results for {len(model_results)} models")

    # Create a LaTeX table
    latex_table = create_latex_table(model_results)

    # Save to file
    output_file = 'mteb_results_table.tex'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(latex_table)

    print(f"\n✓ LaTeX table saved to '{output_file}'")
    print("\n" + "=" * 80)
    print("LaTeX Table Preview:")
    print("=" * 80)
    print(latex_table)
    print("=" * 80)

    # Create Classification-only LaTeX table (per-task columns)
    print("\nCreating classification-only table (per-task) ...")
    clf_results = load_classification_task_results('results')
    latex_table_clf = create_classification_table(clf_results)

    output_file_clf = 'mteb_results_table_clf.tex'
    with open(output_file_clf, 'w', encoding='utf-8') as f:
        f.write(latex_table_clf)
    print(f"\n✓ LaTeX classification table saved to '{output_file_clf}'")
    print("\n" + "=" * 80)
    print("Classification LaTeX Table Preview:")
    print("=" * 80)
    print(latex_table_clf)
    print("=" * 80)

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
