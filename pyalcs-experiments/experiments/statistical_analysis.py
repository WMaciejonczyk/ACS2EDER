import os
import glob
import numpy as np
from scipy.stats import studentized_range
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def analyze_exploit_phase(data_dir="results", target_metric="knowledge"):
    # 1. Znalezienie wszystkich podfolderów (np. 'maze', 'multiplexer') w data_dir
    if not os.path.exists(data_dir):
        print(f"Błąd: Katalog bazowy '{data_dir}' nie istnieje.")
        return

    categories = [d for d in os.listdir(data_dir) if
                  os.path.isdir(os.path.join(data_dir, d))]

    if not categories:
        print(
            f"Nie znaleziono podkatalogów (kategorii) w folderze '{data_dir}'.")
        return

    # 2. Przetwarzanie każdej kategorii z osobna
    for category in categories:
        print(f"\nAnaliza kategorii: {category} (metryka: {target_metric})...")
        category_dir = os.path.join(data_dir, category)

        all_files = glob.glob(os.path.join(category_dir, "res_*_*.csv"))

        if not all_files:
            print(f" -> Nie znaleziono plików CSV w {category_dir}. Pomijam.")
            continue

        full_df = pd.concat((pd.read_csv(f) for f in all_files),
                            ignore_index=True)

        # SPRAWDZENIE: Czy szukana kolumna istnieje w tej kategorii?
        if target_metric not in full_df.columns:
            print(
                f" -> Pomijam: Kolumna '{target_metric}' nie istnieje w danych dla kategorii '{category}'.")
            continue

        # 3. Odfiltrowanie TYLKO fazy exploit
        exploit_df = full_df[full_df['phase'] == 'exploit'].copy()

        if exploit_df.empty:
            print(
                f" -> Błąd: W kategorii '{category}' brak wierszy, gdzie phase == 'exploit'.")
            continue

        environments = exploit_df['env'].unique()

        # 4. Dynamiczne tworzenie folderów na wyniki (np. statistics/maze)
        output_dir = os.path.join("statistics", category, target_metric)
        os.makedirs(output_dir, exist_ok=True)

        for env in environments:
            env_data = exploit_df[exploit_df['env'] == env]

            # ---------------------------------------------------------
            # CZĘŚĆ 1: TABELA ŚREDNICH I ODCHYLEŃ
            # ---------------------------------------------------------
            summary = env_data.groupby('model')[target_metric].agg(
                ['mean', 'std']).round(3)
            summary = summary.sort_values(by='mean', ascending=False)

            # Zapis tabeli do CSV
            csv_filename = os.path.join(output_dir,
                                        f"summary_{target_metric}_{env}.csv")
            summary.to_csv(csv_filename)

        print(
            f" -> Pomyślnie zapisano pliki ({target_metric}) dla {len(environments)} środowisk w '{output_dir}'.")


def generate_friedman_ranking_table(stats_dir="statistics",
                                    target_metric="knowledge",
                                    higher_is_better=True):
    if not os.path.exists(stats_dir):
        print(f"Błąd: Katalog bazowy '{stats_dir}' nie istnieje.")
        return

    categories = [d for d in os.listdir(stats_dir) if
                  os.path.isdir(os.path.join(stats_dir, d))]

    if not categories:
        print(f"Nie znaleziono podkatalogów (kategorii) w folderze '{stats_dir}'.")
        return

    for category in categories:
        print(f"\nGenerowanie rankingu dla kategorii: {category} (metryka: {target_metric})...")

        metric_dir = os.path.join(stats_dir, category, target_metric)

        if not os.path.exists(metric_dir):
            print(f" -> Brak folderu '{target_metric}' w kategorii '{category}'. Pomijam.")
            continue

        file_pattern = os.path.join(metric_dir, f"summary_{target_metric}_*.csv")
        all_files = glob.glob(file_pattern)

        if not all_files:
            print(f" -> Brak plików .csv w folderze {metric_dir}. Pomijam.")
            continue

        dict_mean = {}
        dict_std = {}
        skipped_envs = 0

        for file in all_files:
            env_name = os.path.basename(file).replace(
                f"summary_{target_metric}_", "").replace(".csv", "")

            df = pd.read_csv(file)

            if not df['model'].str.contains("VCP", na=False).any():
                skipped_envs += 1
                continue

            # Pobieramy oddzielnie mean i std
            dict_mean[env_name] = dict(zip(df['model'], df['mean']))
            dict_std[env_name] = dict(zip(df['model'], df['std']))

        if skipped_envs > 0:
            print(f" -> Odrzucono {skipped_envs} środowisk (brak wyników dla VCP). Do analizy wzięto {len(dict_mean)}.")

        if not dict_mean:
            print(" -> Błąd: Po filtracji VCP nie zostało żadne środowisko! Pomijam tworzenie tabeli.")
            continue

        df_mean = pd.DataFrame.from_dict(dict_mean, orient='index')
        df_std = pd.DataFrame.from_dict(dict_std, orient='index')

        setup = [
            'ACS2',
            'ACS2ER (s=8)',
            'ACS2EDER (s=3,b=2)',
            'ACS2EDER (s=5,b=2)',
            'ACS2EDER (s=3,b=4)',
            'ACS2EDER (s=5,b=4)',
            'ACS2HER (s=8,k=2)',
            'ACS2VCP (s=8,k=2,h=4)',
        ]

        df_mean = df_mean.reindex(columns=setup)
        df_std = df_std.reindex(columns=setup)

        df_ranks = df_mean.rank(axis=1, ascending=(not higher_is_better), method='average')

        df_formatted = pd.DataFrame(index=df_mean.index, columns=df_mean.columns)

        for col in df_mean.columns:
            for idx in df_mean.index:
                val_mean = df_mean.loc[idx, col]
                val_std = df_std.loc[idx, col]

                if pd.isna(val_mean):
                    df_formatted.loc[idx, col] = "-"
                else:
                    df_formatted.loc[idx, col] = f"{val_mean:.3f} ({val_std:.3f})"

        avg_means = df_mean.mean()
        df_formatted.loc['AVG'] = [f"{val:.3f}" for val in avg_means]

        average_ranks = df_ranks.mean()
        df_formatted.loc['Nemenyi rank'] = [f"{rank:.3f}" for rank in average_ranks]

        output_filename = os.path.join(metric_dir, f"FINAL_Friedman_{target_metric}.csv")
        df_formatted.to_csv(output_filename)

        latex_filename = os.path.join(metric_dir,
                                      f"FINAL_Friedman_{target_metric}.tex")

        df_formatted.to_latex(
            latex_filename,
            column_format='l' + 'c' * len(df_formatted.columns),
            na_rep='-',
            escape=False
        )
        print(f"Zapisano do pliku: {output_filename}")

        print(f"Zapisano tabelę LaTeX do: {latex_filename}")


def generate_dual_friedman_table(stats_dir="statistics",
                                 metric1="numerosity",
                                 metric2="reliable",
                                 m1_higher_is_better=False,
                                 m2_higher_is_better=True):

    if not os.path.exists(stats_dir):
        print(f"Błąd: Katalog bazowy '{stats_dir}' nie istnieje.")
        return

    categories = [d for d in os.listdir(stats_dir) if
                  os.path.isdir(os.path.join(stats_dir, d))]

    setup = [
        'ACS2',
        'ACS2ER (s=8)',
        'ACS2EDER (s=3,b=2)',
        'ACS2EDER (s=5,b=2)',
        'ACS2EDER (s=3,b=4)',
        'ACS2EDER (s=5,b=4)',
        'ACS2HER (s=8,k=2)',
        'ACS2VCP (s=8,k=2,h=4)',
    ]

    for category in categories:

        m1_dir = os.path.join(stats_dir, category, metric1)
        m2_dir = os.path.join(stats_dir, category, metric2)

        if not os.path.exists(m1_dir) or not os.path.exists(m2_dir):
            print(
                f" -> Brak jednego z folderów metryk w '{category}'. Pomijam.")
            continue

        m1_files = glob.glob(os.path.join(m1_dir, f"summary_{metric1}_*.csv"))

        if not m1_files:
            continue

        dict_m1 = {}
        dict_m2 = {}

        for f1 in m1_files:
            env_name = os.path.basename(f1).replace(f"summary_{metric1}_",
                                                    "").replace(".csv", "")
            f2 = os.path.join(m2_dir, f"summary_{metric2}_{env_name}.csv")

            if not os.path.exists(f2):
                continue

            df1 = pd.read_csv(f1)
            df2 = pd.read_csv(f2)

            if not df1['model'].str.contains("VCP", na=False).any():
                continue

            dict_m1[env_name] = dict(zip(df1['model'], df1['mean']))
            dict_m2[env_name] = dict(zip(df2['model'], df2['mean']))

        if not dict_m1:
            print(" -> Błąd: Brak wspólnych środowisk z VCP. Pomijam.")
            continue

        df_m1 = pd.DataFrame.from_dict(dict_m1, orient='index')
        df_m2 = pd.DataFrame.from_dict(dict_m2, orient='index')

        df_m1 = df_m1.reindex(columns=setup)
        df_m2 = df_m2.reindex(columns=setup)

        df_ranks_m1 = df_m1.rank(axis=1, ascending=(not m1_higher_is_better),
                                 method='average')
        df_ranks_m2 = df_m2.rank(axis=1, ascending=(not m2_higher_is_better),
                                 method='average')

        avg_m1 = df_m1.mean()
        avg_m2 = df_m2.mean()
        ranks_m1 = df_ranks_m1.mean()
        ranks_m2 = df_ranks_m2.mean()

        models = df_m1.columns
        columns = pd.MultiIndex.from_product([models, ['Num', 'Rel']],
                                             names=['Model', 'Metric'])
        df_final = pd.DataFrame(index=df_m1.index, columns=columns)

        for col in models:
            df_final[(col, 'Num')] = df_m1[col].map('{:.2f}'.format)
            df_final[(col, 'Rel')] = df_m2[col].map('{:.2f}'.format)

        avg_row = []
        for col in models:
            avg_row.extend([f"{avg_m1[col]:.2f}", f"{avg_m2[col]:.2f}"])
        df_final.loc['AVG'] = avg_row

        rank_m1_row = []
        for col in models:
            rank_m1_row.extend([f"{ranks_m1[col]:.3f}", ""])
        df_final.loc['Nemenyi rank (num)'] = rank_m1_row

        rank_m2_row = []
        for col in models:
            rank_m2_row.extend(["", f"{ranks_m2[col]:.3f}"])
        df_final.loc['Nemenyi rank (rel)'] = rank_m2_row

        output_csv = os.path.join(stats_dir, category,
                                  f"FINAL_Dual_{metric1}_{metric2}.csv")
        df_final.to_csv(output_csv)

        latex_filename = os.path.join(stats_dir, category,
                                      f"FINAL_Dual_{metric1}_{metric2}.tex")
        df_final.to_latex(
            latex_filename,
            column_format='l' + 'c' * len(df_final.columns),
            na_rep='-',
            escape=False,
            multicolumn=True,
            multicolumn_format='c'
        )

        print("-" * 50)
        print("Tabela podwójna wygenerowana z sukcesem!")
        print(f"Zapisano do pliku: {output_csv}")


def calculate_nemenyi_cd(k, N, alpha=0.05):
    q_val = studentized_range.ppf(1 - alpha, k, np.inf)

    q_alpha = q_val / np.sqrt(2)

    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * N))
    return cd


def generate_critical_difference_diagram(metric, cd_value):
    path = f"statistics/maze/{metric}/FINAL_Friedman_{metric}.csv"
    df = pd.read_csv(path, index_col=0)
    nemenyi_ranks = df.iloc[-1].astype(float)

    models = nemenyi_ranks.index
    sig_matrix = pd.DataFrame(1.0, index=models, columns=models)

    for m1 in models:
        for m2 in models:
            if abs(nemenyi_ranks[m1] - nemenyi_ranks[m2]) > cd_value:
                sig_matrix.loc[m1, m2] = 0.01

    plt.figure(figsize=(12, 5))
    sp.critical_difference_diagram(nemenyi_ranks, sig_matrix)
    output_dir = os.path.join("plots", "maze", "nemenyi")
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"plot_Nemenyi_{metric}.pdf")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')


def generate_split_cd_diagrams(filepath):
    df = pd.read_csv(filepath, index_col=0, header=None)

    df.index = df.index.astype(str).str.strip()

    models_row = df.loc['Model']
    metrics_row = df.loc['Metric']

    start_idx = df.index.get_loc('Metric') + 1
    end_idx = df.index.get_loc('AVG')
    N = end_idx - start_idx

    output_dir = os.path.join("plots", "maze", "nemenyi")
    os.makedirs(output_dir, exist_ok=True)

    for target_metric, rank_row_name in [('Num', 'Nemenyi rank (num)'),
                                         ('Rel', 'Nemenyi rank (rel)')]:

        valid_cols = metrics_row[
            metrics_row.str.strip() == target_metric].index

        models = models_row[valid_cols].values
        ranks = df.loc[rank_row_name, valid_cols].astype(float).values

        nemenyi_ranks = pd.Series(ranks, index=models)

        k = len(models)
        cd_value = calculate_nemenyi_cd(k, N)

        print(f"--- Metryka: {target_metric} ---")
        print(f"Liczba modeli (k) = {k}, Liczba zbiorów (N) = {N}")
        print(f"Wyliczone CD = {cd_value:.4f}\n")

        sig_matrix = pd.DataFrame(1.0, index=models, columns=models)
        for m1 in models:
            for m2 in models:
                if abs(nemenyi_ranks[m1] - nemenyi_ranks[m2]) > cd_value:
                    sig_matrix.loc[m1, m2] = 0.01

        plt.figure(figsize=(12, 5))

        sp.critical_difference_diagram(nemenyi_ranks, sig_matrix)

        output_filename = os.path.join(output_dir,
                                       f"plot_Nemenyi_{target_metric}.pdf")
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    # analyze_exploit_phase(data_dir="results", target_metric="knowledge")
    # analyze_exploit_phase(data_dir="results", target_metric="reward")
    # analyze_exploit_phase(data_dir="results", target_metric="steps_in_trial")
    # analyze_exploit_phase(data_dir="results", target_metric="numerosity")
    # analyze_exploit_phase(data_dir="results", target_metric="reliable")
    # generate_friedman_ranking_table(target_metric="knowledge", higher_is_better=True)
    # generate_friedman_ranking_table(target_metric="reward", higher_is_better=True)
    # generate_friedman_ranking_table(target_metric="steps_in_trial", higher_is_better=False)
    # generate_dual_friedman_table()
    cd_value = calculate_nemenyi_cd(k=8, N=27, alpha=0.05)
    print(f"Critical Difference (CD) for k=8, N=27, alpha=0.05: {cd_value:.3f}")

    # generate_critical_difference_diagram(metric="knowledge", cd_value=cd_value)
    # generate_critical_difference_diagram(metric="steps_in_trial", cd_value=cd_value)

    path = "statistics/maze/FINAL_Dual_numerosity_reliable.csv"
    generate_split_cd_diagrams(path)
