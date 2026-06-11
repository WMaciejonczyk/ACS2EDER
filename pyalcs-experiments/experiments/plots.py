import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import numpy as np

def load_data(data_dir="results"):
    """Wczytuje i łączy wszystkie pliki CSV tylko raz."""
    file_pattern = os.path.join(data_dir, "res_*_*.csv")
    all_files = glob.glob(file_pattern)

    if not all_files:
        print("Nie znaleziono żadnych plików CSV.")
        return None

    df_list = [pd.read_csv(f) for f in all_files]
    full_df = pd.concat(df_list, ignore_index=True)

    return full_df


def plot_single_metric(full_df, metric_col, folder_name, y_label):
    """Uniwersalna funkcja do rysowania pojedynczych metryk (knowledge, reward, steps)."""
    sns.set_theme(style="whitegrid")
    environments = full_df['env'].unique()

    for env in environments:
        plt.figure(figsize=(10, 6))
        env_data = full_df[full_df['env'] == env]

        avg_df = env_data.groupby(['model', 'trial'])[
            metric_col].mean().reset_index()
        avg_df = avg_df.sort_values(by=['model', 'trial'])

        if 'multiplexer' in env:
            avg_df['boundary'] = 5000
        elif 'corridor' in env:
            avg_df['boundary'] = 500
        else:
            condition = avg_df['model'].isin(
                ['ACS2HER (s=8,k=2)', 'ACS2VCP (s=8,k=2,h=4)'])
            avg_df['boundary'] = np.where(condition, 500, 5000)

        avg_df['is_exploit'] = avg_df['trial'] > avg_df['boundary']

        smooth_col = f'{metric_col}_smooth'
        avg_df['is_exploit'] = avg_df['trial'] > 5000
        avg_df[smooth_col] = avg_df.groupby(['model', 'is_exploit'])[
            metric_col].transform(
            lambda x: x.rolling(window=50, min_periods=1).mean()
        )
        sns.lineplot(
            data=avg_df, x='trial', y=smooth_col,
            hue='model', style='model', markers=True, dashes=False,
            markevery=100, linewidth=0.8, markersize=6, errorbar="sd"
        )

        unique_boundaries = sorted(avg_df['boundary'].unique())
        for b in unique_boundaries:
            if b == 500 and 'corridor' in env:
                label_str = 'explore/exploit'
                color = 'black'
            elif b == 500:
                label_str = 'explore/exploit - ACS2HER, ACS2VCP'
                color = 'gray'
            else:
                label_str = 'explore/exploit - Inne modele' if 500 in unique_boundaries else 'explore/exploit'
                color = 'black'

            plt.axvline(x=b, color=color, linestyle='--', linewidth=1.2,
                        alpha=0.8, label=label_str)

        plt.xlabel('Trial', fontsize=12)
        plt.ylabel(y_label, fontsize=12)

        plt.legend(frameon=True, fontsize=12)
        plt.tight_layout()

        if 'multiplexer' in env:
            output_dir = os.path.join("plots", "multiplexer", folder_name)
            os.makedirs(output_dir, exist_ok=True)
        elif 'corridor' in env:
            output_dir = os.path.join("plots", "corridor", folder_name)
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = os.path.join("plots", "maze", folder_name)
            os.makedirs(output_dir, exist_ok=True)

        output_filename = os.path.join(output_dir,
                                       f"plot_{folder_name}_{env}.pdf")
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Zapisano wykres: {output_filename}")

        plt.close()


def plot_combined_num_rel(full_df):
    """Funkcja do rysowania wykresu łączącego numerosity i reliable."""
    sns.set_theme(style="whitegrid")
    environments = full_df['env'].unique()

    for env in environments:
        plt.figure(figsize=(10, 6))
        env_data = full_df[full_df['env'] == env]

        avg_df = env_data.groupby(['model', 'trial'])[
            ['numerosity', 'reliable']].mean().reset_index()
        avg_df = avg_df.sort_values(by=['model', 'trial'])

        if 'multiplexer' in env:
            avg_df['boundary'] = 5000
        elif 'corridor' in env:
            avg_df['boundary'] = 500
        else:
            condition = avg_df['model'].isin(
                ['ACS2HER (s=8,k=2)', 'ACS2VCP (s=8,k=2,h=4)'])
            avg_df['boundary'] = np.where(condition, 500, 5000)

        avg_df['is_exploit'] = avg_df['trial'] > avg_df['boundary']

        avg_df['num_smooth'] = avg_df.groupby(['model', 'is_exploit'])[
            'numerosity'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
        avg_df['rel_smooth'] = avg_df.groupby(['model', 'is_exploit'])[
            'reliable'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )

        melted_df = pd.melt(
            avg_df, id_vars=['model', 'trial'],
            value_vars=['num_smooth', 'rel_smooth'],
            var_name='metric_type', value_name='value'
        )

        melted_df['legend_label'] = melted_df.apply(
            lambda row: f"{row['model']} (num)" if row[
                                                       'metric_type'] == 'num_smooth' else f"{row['model']} (rel)",
            axis=1
        )

        sns.lineplot(
            data=melted_df, x='trial', y='value',
            hue='legend_label', style='legend_label', markers=True,
            dashes=False,
            markevery=100, linewidth=0.8, markersize=6
        )

        unique_boundaries = sorted(avg_df['boundary'].unique())
        for b in unique_boundaries:
            if b == 500 and 'corridor' in env:
                label_str = 'explore/exploit'
                color = 'black'
            elif b == 500:
                label_str = 'explore/exploit - ACS2HER, ACS2VCP'
                color = 'gray'
            else:
                label_str = 'explore/exploit - Inne modele' if 500 in unique_boundaries else 'explore/exploit'
                color = 'black'

            plt.axvline(x=b, color=color, linestyle='--', linewidth=1.2,
                        alpha=0.8, label=label_str)

        plt.xlabel('Trial', fontsize=12)
        plt.ylabel('Classifiers Count', fontsize=12)

        plt.legend(frameon=True, fontsize=9)
        plt.tight_layout()

        if 'multiplexer' in env:
            output_dir = os.path.join("plots", "multiplexer", "num_rel")
            os.makedirs(output_dir, exist_ok=True)
        elif 'corridor' in env:
            output_dir = os.path.join("plots", "corridor", "num_rel")
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = os.path.join("plots", "maze", "num_rel")
            os.makedirs(output_dir, exist_ok=True)

        output_filename = os.path.join(output_dir, f"plot_num_rel_{env}.pdf")
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Zapisano wykres: {output_filename}")

        plt.close()


if __name__ == "__main__":
    df = load_data("results/corridor")

    if df is not None:
        # plot_single_metric(df, metric_col='knowledge', folder_name='knowledge',
        #                    y_label='Knowledge [%]')
        # plot_single_metric(df, metric_col='reward', folder_name='reward',
        #                    y_label='Reward')
        plot_single_metric(df, metric_col='steps_in_trial',
                           folder_name='steps', y_label='Steps')
        plot_combined_num_rel(df)
