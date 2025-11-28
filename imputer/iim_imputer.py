import time
import pandas as pd
import re
import numpy
import numpy as np
from typing import List
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge


class IIMImputer:
    """
    IIM填补方法
    """

    def __init__(self, number_neighbor=10, algo_code="iim 2"):
        self.number_neighbor = number_neighbor
        self.algo_code = algo_code

    def impute(self, incomp_data, verbose=True):
        """
        执行IIM填补
        """
        start_time = time.time()

        recov_data = self.impute_with_algorithm(self.algo_code, incomp_data.copy(), self.number_neighbor,
                                                verbose=verbose)

        end_time = time.time()
        if verbose:
            print(f"\n> logs: imputation iim - Execution Time: {(end_time - start_time):.4f} seconds\n")

        return recov_data

    def iim_recovery(self, matrix_nan: np.ndarray, adaptive_flag: bool = False, learning_neighbors: int = 10):
        """Implementation of the IIM algorithm"""
        tuples_with_nan = np.isnan(matrix_nan).any(axis=1)
        if np.any(tuples_with_nan):
            incomplete_tuples_indices = np.array(np.where(tuples_with_nan))
            incomplete_tuples = matrix_nan[tuples_with_nan]
            complete_tuples = matrix_nan[~tuples_with_nan]

            if not isinstance(learning_neighbors, (int, numpy.int64)):
                learning_neighbors = learning_neighbors[0]

            if learning_neighbors > len(complete_tuples):
                learning_neighbors = min(len(complete_tuples), learning_neighbors)
            if len(complete_tuples) == 0:
                return matrix_nan

            if adaptive_flag:
                lr_models = self.adaptive(complete_tuples, incomplete_tuples, learning_neighbors,
                                          max_learning_neighbors=min(len(complete_tuples), 10))
                imputation_result = self.imputation(incomplete_tuples, lr_models)
            else:
                lr_models = self.learning(complete_tuples, incomplete_tuples, learning_neighbors)
                imputation_result = self.imputation(incomplete_tuples, lr_models)

            for result in imputation_result:
                matrix_nan[np.array(incomplete_tuples_indices)[:, result[0]], result[1]] = result[2]
            return matrix_nan
        else:
            return matrix_nan

    def learning(self, complete_tuples: np.ndarray, incomplete_tuples: np.ndarray, learn: int = 10):
        """Algorithm 1: Learning"""
        knn_euc = NearestNeighbors(n_neighbors=learn, metric='euclidean').fit(complete_tuples)
        number_of_attributes = incomplete_tuples.shape[1]
        model_params = np.empty((len(incomplete_tuples), number_of_attributes, learn), dtype=object)

        incomplete_tuples_no_nan = np.nan_to_num(incomplete_tuples)
        learning_neighbors = knn_euc.kneighbors(incomplete_tuples_no_nan, return_distance=False)

        for tuple_index, incomplete_tuple in enumerate(incomplete_tuples):
            nan_indicator = np.isnan(incomplete_tuple)

            if np.count_nonzero(nan_indicator) == 1:
                nan_index = np.where(nan_indicator)[0][0]
                X = complete_tuples[learning_neighbors[tuple_index]][:, ~nan_indicator]
                y = complete_tuples[learning_neighbors[tuple_index]][:, nan_indicator]
                models = [Ridge(tol=1e-20).fit(X_i.reshape(1, -1), y_i) for X_i, y_i in zip(X, y)]
                model_params[tuple_index, nan_index] = [(model.coef_, model.intercept_) for model in models]
            else:
                for missing_value_index in np.where(nan_indicator)[0]:
                    current_nan_indicator = np.zeros_like(nan_indicator)
                    current_nan_indicator[missing_value_index] = True
                    X = complete_tuples[learning_neighbors[tuple_index]][:, ~current_nan_indicator]
                    y = complete_tuples[learning_neighbors[tuple_index]][:, current_nan_indicator]
                    models = [Ridge(tol=1e-20).fit(X_i.reshape(1, -1), y_i) for X_i, y_i in zip(X, y)]
                    model_params[tuple_index, missing_value_index] = [(model.coef_, model.intercept_) for model in
                                                                      models]

        return model_params

    def imputation(self, incomplete_tuples: np.ndarray, lr_coef_and_threshold: np.ndarray):
        """Algorithm 2: Imputation"""
        imputed_values = []

        for i, incomplete_tuple in enumerate(incomplete_tuples):
            nan_indicator = np.isnan(incomplete_tuple)

            if np.count_nonzero(nan_indicator) == 1:
                missing_attributes_indices = np.where(nan_indicator)[0]
                incomplete_tuple_no_nan = incomplete_tuple[~nan_indicator]

                for missing_attribute_index in missing_attributes_indices:
                    candidate_suggestions = np.array([coef @ incomplete_tuple_no_nan + intercept for coef, intercept in
                                                      lr_coef_and_threshold[i, missing_attribute_index]])
                    distances = self.compute_distances(candidate_suggestions)
                    weights = self.compute_weights(distances)
                    impute_result = np.sum(candidate_suggestions * weights)
                    imputed_values.append([i, missing_attribute_index, impute_result])
            else:
                missing_attributes_indices = np.where(nan_indicator)[0]
                for missing_value_index in np.where(nan_indicator)[0]:
                    current_nan_indicator = np.zeros_like(nan_indicator)
                    current_nan_indicator[missing_value_index] = True
                    incomplete_tuple_no_nan = np.nan_to_num(incomplete_tuple[~current_nan_indicator])

                    for missing_attribute_index in missing_attributes_indices:
                        candidate_suggestions = np.array(
                            [coef @ incomplete_tuple_no_nan + intercept for coef, intercept in
                             lr_coef_and_threshold[i, missing_attribute_index]])
                        distances = self.compute_distances(candidate_suggestions)
                        weights = self.compute_weights(distances)
                        impute_result = np.sum(candidate_suggestions * weights)
                        imputed_values.append([i, missing_attribute_index, impute_result])

        return imputed_values

    def adaptive(self, complete_tuples: np.ndarray, incomplete_tuples: np.ndarray, k: int,
                 max_learning_neighbors: int = 100, step_size: int = 4):
        """Algorithm 3: Adaptive"""
        all_entries = min(int(complete_tuples.shape[0]), max_learning_neighbors)
        phi_list = [self.learning(complete_tuples, incomplete_tuples, l_learning)
                    for l_learning in range(1, all_entries + 1, step_size)]

        nn = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(complete_tuples)
        number_of_models = max(len(phi_list) - 1, 1)
        number_of_incomplete_tuples = len(incomplete_tuples)
        costs = np.zeros((number_of_incomplete_tuples, number_of_models))

        for log, complete_tuple in enumerate(complete_tuples, 1):
            neighbors = nn.kneighbors(complete_tuple.reshape(1, -1), return_distance=False)[0]
            for incomplete_tuple_idx, incomplete_tuple in enumerate(incomplete_tuples):
                nan_indicator = np.isnan(incomplete_tuple)
                neighbors_filtered = np.delete(complete_tuples[neighbors], nan_indicator, axis=1)
                for my_model in range(0, number_of_models):
                    model_params_for_tuple = phi_list[my_model][incomplete_tuple_idx]
                    for attribute_index, model_params in enumerate(model_params_for_tuple):
                        if model_params is not None and not np.any(model_params is None):
                            coefs, intercepts = zip(*model_params)
                            expanded_coef = np.array(coefs)
                            neighbors_filtered_copy = np.nan_to_num(neighbors_filtered)
                            phi_models = (expanded_coef @ neighbors_filtered_copy[:, :, None]).squeeze() + np.array(
                                intercepts)
                            errors = np.abs(complete_tuple[attribute_index] - phi_models)
                            costs[incomplete_tuple_idx, my_model] += np.sum(np.power(errors, 2)) / len(phi_models)

        best_models_indices = np.argmin(costs, axis=1)
        number_of_attributes = incomplete_tuples.shape[1]
        lr_models = np.empty((number_of_incomplete_tuples, number_of_attributes, number_of_models), dtype=object)

        for i in range(number_of_incomplete_tuples):
            best_models_indices_for_tuple = best_models_indices[i]
            lr_models[i] = phi_list[best_models_indices_for_tuple][i]

        return lr_models

    def compute_distances(self, candidate_suggestions: np.ndarray):
        """Calculate the sum of distances to all other candidates"""
        distances = []
        for i in range(len(candidate_suggestions)):
            temp_distances = np.abs(candidate_suggestions[i] - np.delete(candidate_suggestions, i))
            distances.append(np.sum(temp_distances))
        return distances

    def compute_weights(self, distances: List[float]):
        """Compute weights for candidates"""
        distances_n = np.array(distances)
        weights = np.zeros(distances_n.shape)

        nonzero_indices = distances_n != 0
        weights[nonzero_indices] = 1 / distances_n[nonzero_indices] / np.sum(1 / distances_n[nonzero_indices])

        if np.sum(weights) == 0:
            weights = np.ones(distances_n.shape) / len(distances_n)

        return weights

    def impute_with_algorithm(self, alg_code: str, matrix: np.ndarray, neighbors=None, verbose=True):
        """Imputes the input matrix with a specified algorithm"""
        if verbose:
            print(f"(IMPUTATION) IIM\n\tMatrix: {matrix.shape}\n\talg_code: {alg_code}\n\tneighbors: {neighbors}")

        alg_code_spl = alg_code.split()

        if len(alg_code_spl) > 1:
            match = re.match(r"(\d+)([a-zA-Z]+)", alg_code_spl[1], re.I)
            if match:
                if neighbors is None:
                    neighbors, adaptive_flag = match.groups()
                else:
                    _, adaptive_flag = match.groups()

                matrix_imputed = self.iim_recovery(matrix, adaptive_flag=adaptive_flag.startswith("a"),
                                                   learning_neighbors=int(neighbors))
            else:
                if neighbors is None:
                    neighbors = int(alg_code_spl[1])
                matrix_imputed = self.iim_recovery(matrix, adaptive_flag=False, learning_neighbors=neighbors)
        else:
            matrix_imputed = self.iim_recovery(matrix, adaptive_flag=False, learning_neighbors=self.number_neighbor)

        nan_mask = np.isnan(matrix_imputed)
        matrix_imputed[nan_mask] = np.sqrt(np.finfo('d').max / 100000.0)

        return matrix_imputed


# 为了向后兼容，保留函数
def iim_impute(incomp_data, number_neighbor=10, algo_code="iim 2", logs=True, verbose=True):
    imputer = IIMImputer(number_neighbor, algo_code)
    return imputer.impute(incomp_data, verbose)