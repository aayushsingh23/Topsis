# topsis_cli.py
import pandas as pd
import numpy as np
import argparse

def normalize(matrix):
    norm_matrix = matrix / np.sqrt((matrix**2).sum(axis=0))
    return norm_matrix

def apply_weights(matrix, weights):
    return matrix * weights

def ideal_sol(matrix, is_benefit):
    ideal = matrix.max() if is_benefit else matrix.min()
    negative_ideal = matrix.min() if is_benefit else matrix.max()
    return ideal, negative_ideal

def calculate_topsis_scores(matrix, weights, is_benefit_criteria):
    normalized_matrix = normalize(matrix)
    weighted_matrix = apply_weights(normalized_matrix, weights)
    ideal_solutions = [ideal_sol(weighted_matrix.iloc[:, i], is_benefit_criteria[i])
                       for i in range(weighted_matrix.shape[1])]
    dist_to_ideal = np.sqrt(((weighted_matrix - [ideal[0] for ideal in ideal_solutions])**2).sum(axis=1))
    dist_to_negative_ideal = np.sqrt(((weighted_matrix - [ideal[1] for ideal in ideal_solutions])**2).sum(axis=1))

    topsis_scores = dist_to_negative_ideal / (dist_to_ideal + dist_to_negative_ideal)
    return topsis_scores

def main():
    parser = argparse.ArgumentParser(description="Calculate TOPSIS scores and ranks for a dataset.")
    parser.add_argument("input_file", help="Path to the input Excel file")
    parser.add_argument("output_file", help="Path to save the output CSV file")
    parser.add_argument("--weights", nargs='+', type=float, required=True,
                        help="Weights for each criterion, separated by spaces")
    parser.add_argument("--criteria", nargs='+', type=str, required=True,
                        help="Criteria for each column (benefit or cost), separated by spaces")
    
    args = parser.parse_args()

    df = pd.read_excel(args.input_file)

    criteria_columns = df.columns[1:]
    matrix = df[criteria_columns]
    

    is_benefit_criteria = [True if c.lower() == 'benefit' else False for c in args.criteria]


    topsis_scores = calculate_topsis_scores(matrix, np.array(args.weights), is_benefit_criteria)
    df['TOPSIS Score'] = topsis_scores
    df['Rank'] = df['TOPSIS Score'].rank(ascending=False)


    df.to_csv(args.output_file, index=False)
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
