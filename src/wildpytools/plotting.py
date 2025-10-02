from .data import remove_rare_classes
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly_express as px

def build_cf_matrix(targ_df, 
                    pred_df, 
                    cf_norm_pth=None, 
                    cf_pth=None,
                    cf_fig_pth=None,
                    save_destn=None):
    targ_df, pred_df = remove_rare_classes(targ_df, pred_df, 1)
    target_arr = targ_df.to_numpy()
    predicts_arr = pred_df.to_numpy()
    target_by_row = np.argmax(target_arr, axis=1)
    predict_by_row = np.argmax(predicts_arr, axis=1)
    cols = targ_df.columns
    target_names = [cols[idx] for idx in target_by_row]
    predict_names = [cols[idx] for idx in predict_by_row]
    classes = cols.tolist()
    cf_matrix_raw = confusion_matrix(target_names, predict_names)
    cf_matrix_norm = confusion_matrix(target_names, predict_names, normalize='true')
    df_cm_norm = pd.DataFrame(cf_matrix_norm, index = [i for i in classes],
                              columns=[i for i in classes]).round(decimals=3)
    df_cm_raw = pd.DataFrame(cf_matrix_raw, index=[i for i in classes],
                             columns=[i for i in classes])
    df_cm_raw['Total Annotated'] = df_cm_raw[classes].sum(axis=1)
    plt.figure(figsize=(13, 13))
    ax_cm = sns.heatmap(df_cm_norm, annot=False, fmt='.1f', cmap='Greens')  # cmap="crest" for green/blue
    # Ensure all labels are shown
    ax_cm.set_xticks(np.arange(len(classes)) + 0.5)
    ax_cm.set_yticks(np.arange(len(classes)) + 0.5)
    ax_cm.set_xticklabels(classes, rotation=90)
    ax_cm.set_yticklabels(classes, rotation=0)
    ax_cm.set(xlabel="Target", ylabel="Prediction")
    ax_cm.set_title('Confusion Matrix')
    df_cm_norm['Total Annotated'] = df_cm_norm[classes].sum(axis=1)
    df_cm_norm['Total Annotated'] = df_cm_norm['Total Annotated'].round(decimals=0)
    if save_destn is not None:
        plt.savefig(save_destn, format="pdf", bbox_inches="tight")
    if cf_fig_pth:
        plt.savefig(cf_fig_pth)
    if cf_pth:
        df_cm_raw.to_csv(cf_pth)
    if cf_norm_pth:
        df_cm_norm.to_csv(cf_norm_pth)
    return


def plot_scores_by_class(df_target, df_pred, scoring_fn, title=None, min_samples=5, height=1200):
    df_target, df_pred = remove_rare_classes(df_target, df_pred, min_samples)
    score_dict = scoring_fn(df_target, df_pred, average=None)
    col_sums = df_target.sum()
    sorted_cols = col_sums.sort_values(ascending=True)
    names = [name for name in sorted_cols.index]
    counts = [count for count in sorted_cols]
    scores = [score_dict[name] for name in names]
    df = pd.DataFrame({'Species Names': names, 'Counts': counts, 'Scores': scores})
    df["Scores"] = pd.to_numeric(df["Scores"])
    df["Counts"] = pd.to_numeric(df["Counts"])
    colour_map=['#2b1d0e', '#a4907c', "#9DAA65", "#546D22", "#1d5013", '#0d4733']
    fig = px.bar(df,
                 x='Scores',
                 y='Species Names',
                 color='Counts',
                 orientation='h',
                 hover_data=['Counts', 'Scores'],
                 color_continuous_scale=colour_map,
                 range_x=[0, 1])
    if title is not None:
        fig.update_layout(title=title)
    fig.update_layout(height=height, plot_bgcolor='rgb(128, 128, 128)')  # Transparent plot background)
    fig.update_layout(yaxis=dict(tickmode='array', tickvals=df["Species Names"], ticktext=df["Species Names"]))
    fig.show()
    return names, scores


def plot_continuous(df, column_name, x_max=None, x_min=None, bins=None):
    plt.figure(figsize=(6, 4))
    ax = sns.histplot(df[column_name], bins=bins, kde=True)
    plt.title(f'Distribution of {column_name} with {bins} Bins and KDE')
    plt.xlabel(column_name)
    ax.set(xlim=(x_min, x_max) if x_min is not None and x_max is not None else None)
    plt.ylabel('Count')
    plt.show()


def plot_two_distributions(dist1, 
                           dist2, 
                           label1='Distribution 1', 
                           label2='Distribution 2', 
                           x_max=None, 
                           y_max=None,
                           bins=None):
    
    plt.figure(figsize=(6, 4))
    ax = sns.histplot(dist1, bins=bins, kde=False, color="#3b856b", label=label1, stat='density')
    sns.histplot(dist2, bins=bins, kde=False, color="#553d24", label=label2, stat='density', ax=ax)
    plt.title(f'{label1} and {label2} score distributions')
    plt.xlabel('Value')
    ax.set(xlim=(0, x_max) if x_max is not None else None)
    ax.set(ylim=(0, y_max) if y_max is not None else None)
    plt.ylabel('Density')
    plt.legend()
    plt.show()


def predictions_by_class(class_name, df, show_results=False):
    df = df[[class_name, 'Targets']]
    positive_vals = df[df['Targets']==class_name][class_name].values
    negative_vals = df[df['Targets']!=class_name][class_name].values

    positive_mean, negative_mean = 0,0
    if len(positive_vals) >= 1:
        positive_mean = positive_vals.mean()
        negative_mean = negative_vals.mean()
    
    if show_results:
        print(f'The mean for true positives is {positive_mean: 0.3f}')
        print(f'The mean for true negatives is {negative_mean: 0.3f}')
        plot_two_distributions(positive_vals, 
                        negative_vals, 
                        label1=f'True Positive {class_name}', 
                        label2=f'True Negative {class_name}', 
                        bins=40, 
                        x_max=1, 
                        y_max=40)
    return positive_mean, negative_mean