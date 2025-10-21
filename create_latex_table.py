import json
from pathlib import Path
from collections import defaultdict
import numpy as np

# Define MTEB task type mappings
TASK_TYPE_MAPPING = {
    # Retrieval tasks
    'BelebeleRetrieval': 'Retrieval',
    'SlovakSumRetrieval': 'Retrieval',
    'SMESumRetrieval': 'Retrieval',
    'SKQuadRetrieval': 'Retrieval',
    'WebFAQRetrieval': 'Retrieval',

    # Classification tasks
    'SlovakHateSpeechClassification.v2': 'Classification',
    'SlovakMovieReviewSentimentClassification.v2': 'Classification',
    'SIB200Classification': 'Classification',
    'MultilingualSentimentClassification': 'Classification',
    'MultiEURLEXMultilabelClassification': 'Classification',
    'DGurgurovSlovakSentiment': 'Classification',
    'SlovakParlaSentClassification': 'Classification',
    'MultiEupSlovakPartyClassification': 'Classification',
    'MultiEupSlovakGenderClassification': 'Classification',

    # Clustering tasks
    'SIB200ClusteringS2S': 'Clustering',
    'PravdaSKTagClustering': 'Clustering',
    'PravdaSKURLClustering': 'Clustering',
    'SlovakSumURLClustering': 'Clustering',
    'SMESumCategoryClustering': 'Clustering',

    # Reranking tasks
    'SkQuadReranking': 'Reranking',
    'SlovakFactCheckReranking': 'Reranking',

    # STS tasks
    'SlovakSTS': 'STS',

    # Pair Classification
    'SlovakNLI': 'PairClassification',
    'SlovakRTE': 'PairClassification',
    'DemagogSKNLI': 'PairClassification',

    # BitextMining
    'OpusSlovakEnglishBitextMining': 'BitextMining',
    'Tatoeba': 'BitextMining',
    'FloresBitextMining': 'BitextMining',
    'NTREXBitextMining': 'BitextMining',
    'WebFAQBitextMiningQuestions': 'BitextMining',
    'WebFAQBitextMiningQAs': 'BitextMining',
}

SPLITS = set()
def extract_main_score(task_data):
    """Extract the main score from a task result"""
    scores = task_data.get('scores', {})

    for split in scores:
        SPLITS.add(split)
    
    for split in ['test', 'devtest', 'default']:
        if split in scores:
            split_scores = scores[split]
            if isinstance(split_scores, list):
                main_scores = [entry.get('main_score', np.nan) for entry in split_scores]
                return np.nanmean(main_scores) if main_scores else np.nan
            elif isinstance(split_scores, dict):
                return split_scores.get('main_score', np.nan)
        else:
            print(f"Err: Missing split scores for task: {task_data['task_name']}")
    
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
                    
                    main_score = extract_main_score(task_data)
                    task_type = TASK_TYPE_MAPPING.get(task_name, 'Unknown')
                    
                    if not np.isnan(main_score) and task_type != 'Unknown':
                        model_results[model_name][task_type].append(main_score)
                        
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error reading {json_file}: {e}")
    
    return model_results

def format_model_name(model_name):
    """Format model name for LaTeX (escape underscores)"""
    # Replace double underscores with a formatted version
    return model_name.replace('__', '/').replace('_', '\\_')

def create_latex_table(model_results):
    """Create a LaTeX table with models as rows and task type averages as columns"""
    # Prepare data
    table_data = []
    
    for model_name, task_types in model_results.items():
        row = {'Model': format_model_name(model_name)}
        
        # Calculate an average for each task type
        for task_type, scores in task_types.items():
            avg_score = np.mean(scores) * 100  # Convert to percentage
            row[task_type] = avg_score
        
        # Calculate overall average
        all_scores = [score for scores in task_types.values() for score in scores]
        if all_scores:
            row['Average'] = np.mean(all_scores) * 100
        
        table_data.append(row)
    
    # Sort by average score (descending)
    table_data.sort(key=lambda x: x.get('Average', 0), reverse=True)
    
    # Get all task types (columns)
    all_task_types = sorted(set(
        task_type 
        for row in table_data 
        for task_type in row.keys() 
        if task_type not in ['Model', 'Average']
    ))
    
    # Build LaTeX table
    # Build LaTeX table
    latex = list()
    latex.append("\\begin{sidewaystable*}[p]")  # Changed from table* to sidewaystable*
    latex.append("\\centering")
    latex.append("\\caption{MTEB Results Summary - Average Scores by Task Type (\\%)}")
    latex.append("\\label{tab:mteb_results}")
    latex.append("\\small")

    # Column specification
    num_cols = 2 + len(all_task_types)  # Model + Average + task types
    col_spec = "l" + "r" * (num_cols - 1)
    latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex.append("\\toprule")

    # Header
    header = ["\\textbf{Model}", "\\textbf{Avg}"] + [f"\\textbf{{{tt}}}" for tt in all_task_types]
    latex.append(" & ".join(header) + " \\\\")
    latex.append("\\midrule")

    # Determine the best score per task type (for bolding one value per column)
    best_per_type = {}
    for tt in all_task_types:
        col_values = [r.get(tt, np.nan) for r in table_data]
        col_values = [v for v in col_values if not np.isnan(v)]
        best_per_type[tt] = max(col_values) if col_values else np.nan

    # Data rows
    bolded_done = {tt: False for tt in all_task_types}
    for i, row in enumerate(table_data):
        model_name = row['Model']
        avg = row.get('Average', np.nan)

        # Build row
        row_str = f"{model_name} & "

        if not np.isnan(avg):
            row_str += f"{avg:.2f}"
        else:
            row_str += "-"

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
    latex.append("\\end{sidewaystable*}")  # Changed from table* to sidewaystable*

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
