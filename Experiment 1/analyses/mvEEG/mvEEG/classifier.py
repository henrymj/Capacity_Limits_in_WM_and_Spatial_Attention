import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from collections import defaultdict


class Classifier:
    """
    Class to run basic decoding algorithms on EEG data.

    Parameters:
        labels (list): List of integer labels to be used for decoding
        classifier (sklearn classifier), default LogisticRegression: Classifier to be used for decoding
        scaler (sklearn scaler), default StandardScaler: Scaler to be used for decoding
    """

    def __init__(self, labels, classifier=None, scaler=None):
        self.labels = labels
        self.n_labels = len(labels)
        if classifier is not None:
            self.classifier = classifier
        else:
            self.classifier = LogisticRegression()
        if scaler is not None:
            self.scaler = scaler
        else:
            self.scaler = StandardScaler()
        self.rng = np.random.default_rng()

    def _standardize(self, X_train, X_test):
        """
        Helper function to standardize data.
        Standardizes test to mean and std of the training set

        Args:
            X_train (np.array): Training data
            X_test (np.array): Testing data

        """

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        return X_train, X_test

    def _get_acc(self, X_test, y_train, y_test):
        """
        Helper function that computes accuracy, but only for test trials whose labels were present in the training set
        """
        preds = self.classifier.predict(X_test)
        labeled_test = [i for i, y in enumerate(y_test) if y in np.unique(y_train)]
        acc = np.mean(preds[labeled_test] == y_test[labeled_test])
        return acc

    def _get_auc(self, X_test, y_train, y_test):
        """
        Helper function that computes AUC, but only for test trials whose labels were present in the training set
        """

        preds = self.classifier.decision_function(X_test)
        labeled_test = [i for i, y in enumerate(y_test) if y in np.unique(y_train)]
        auc = roc_auc_score(y_test[labeled_test], preds[labeled_test])
        return auc

    def _fit(self, X_train, y_train):
        X_train, _ = self._standardize(X_train, X_train)
        self.classifier.fit(X_train, y_train)

    def _decode(self, X_train, X_test, y_train, y_test, fit=True, measures=None):
        """
        Helper function to return decoding metrics on a single timepoint

        Args:
            X_train (np.ndarray): Training data
            X_test (np.ndarray): Testing data
            y_train (np.ndarray): Training labels
            y_test (np.ndarray): Testing labels
            measures : list of str, default None
                List of measures to compute. If None, all measures are computed.
                Available measures are:
                - 'acc': Classification accuracy on test set for each time point.
                - 'acc_shuff': Accuracy on shuffled test set, empirical chance for each time point.
                - 'conf_mat': Confusion matrix for each condition for each time point.
                - 'auc': Area under the ROC curve for each time point.
                - 'auc_shuff': AUC on shuffled test set, empirical chance for each time point.
                - 'confidence_scores': Confidence scores for each condition present in the test set for each time point.

        Returns:
        results: dict
            Dictionary containing the decoding results for each time point.
            refer to the measures parameter for the keys in the dictionary.

        """

        if measures is None:
            measures = ["acc", "acc_shuff", "auc", "auc_shuff"]

        X_train, X_test = self._standardize(X_train, X_test)
        if fit:
            self.classifier.fit(X_train, y_train)

        results = {}
        if "acc" in measures:
            results["acc"] = self._get_acc(X_test, y_train, y_test)  # get accuracy score
        if "acc_shuff" in measures:
            results["acc_shuff"] = self._get_acc(X_test, y_train, self.rng.permutation(y_test))
        if "conf_mat" in measures:
            results["conf_mat"] = confusion_matrix(y_test, y_pred=self.classifier.predict(X_test), labels=self.labels)
        if "auc" in measures:
            results["auc"] = self._get_auc(X_test, y_train, y_test)
        if "auc_shuff" in measures:
            results["auc_shuff"] = self._get_auc(X_test, y_train, self.rng.permutation(y_test))
        if "confidence_scores" in measures:
            confidence_scores = np.full(self.n_labels, np.nan)
            confidence_scores_all = self.classifier.decision_function(X_test)
            for i, ss in enumerate(self.labels):
                confidence_scores[i] = confidence_scores_all[y_test == ss].mean()
            results["confidence_scores"] = confidence_scores

        return results

    def decode_across_time(self, X_train, X_test, y_train, y_test, **kwargs):
        """
        Main function to run decoding across each timepoint

        Args:
            X_train (np.ndarray, shape (n_bins,n_channels,n_times)): Training data
            X_test (np.ndarray, shape (n_bins,n_channels,n_times)): Testing data
            y_train (np.ndarray, shape (n_bins)): Training labels
            y_test (np.ndarray, shape (n_bins)): Testing labels


        Returns:
            accs (np.ndarray, shape (n_times)): Classification accuracy on test set for each timepoint
            accs_shuff (np.ndarray, shape (n_times)): Accuracy on shuffled test set, empirical chance for each timepoint
            conf_mats (np.ndarray, shape (n_labels,n_labels,n_times, n_times)): Confusion matrix for each condition for each timepoint
            confidence_scores (np.ndarray, shape (n_labels,n_times, n_times)): Confidence scores for each condition present in the test set for each timepoint


        """
        ntimes = X_train.shape[2]

        results = defaultdict(list)
        for itime in range(ntimes):
            curr_res = self._decode(X_train[:, :, itime], X_test[:, :, itime], y_train, y_test, **kwargs)
            for key, val in curr_res.items():
                results[key].append(val)

        # convert to np.arrays
        for key, val in results.items():
            results[key] = np.array(val)

        return results

    def temporally_generalize(self, X_train, X_test, y_train, y_test, measures=None):
        """
        Perform temporal generalization by training and testing a classifier
        across different time points and calculating accuracy, shuffled accuracy,
        confusion matrices, and confidence scores.

        Parameters:
        X_train : ndarray
            Training data of shape (n_samples, n_features, n_times).
        X_test : ndarray
            Testing data of shape (n_samples, n_features, n_times).
        y_train : ndarray
            Labels for the training data of shape (n_samples,).
        y_test : ndarray
            Labels for the testing data of shape (n_samples,).
        measures : list of str, default None
            List of measures to compute. If None, all measures are computed.
            Available measures are:
            - 'acc': Classification accuracy on test set for each time point.
            - 'acc_shuff': Accuracy on shuffled test set, empirical chance for each time point.
            - 'conf_mat': Confusion matrix for each condition for each time point.
            - 'auc': Area under the ROC curve for each time point.
            - 'auc_shuff': AUC on shuffled test set, empirical chance for each time point.
            - 'confidence_scores': Confidence scores for each condition present in the test set for each time point.

        Returns:
        results: dict
            Dictionary containing the decoding results for each time point.
            refer to the measures parameter for the keys in the dictionary.
        """
        ntimes = X_train.shape[2]

        if measures is None:
            measures = ["acc", "acc_shuff", "conf_mat", "auc", "auc_shuff", "confidence_scores"]
        # need to preset the results to avoid differences in dimension size
        results = {}
        if "acc" in measures:
            results["acc"] = np.full((ntimes, ntimes), np.nan)
        if "acc_shuff" in measures:
            results["acc_shuff"] = np.full((ntimes, ntimes), np.nan)
        if "conf_mat" in measures:
            results["conf_mat"] = np.full((ntimes, ntimes, self.n_labels, self.n_labels), np.nan)
        if "auc" in measures:
            results["auc"] = np.full((ntimes, ntimes), np.nan)
        if "auc_shuff" in measures:
            results["auc_shuff"] = np.full((ntimes, ntimes), np.nan)
        if "confidence_scores" in measures:
            results["confidence_scores"] = np.full((ntimes, ntimes, self.n_labels), np.nan)

        for itime in range(ntimes):  # train times
            self._fit(X_train[:, :, itime], y_train)
            for jtime in range(ntimes):  # test times
                curr_res = self._decode(
                    X_train[:, :, itime], X_test[:, :, jtime], y_train, y_test, fit=False, measures=measures
                )
                for key, val in curr_res.items():
                    results[key][itime, jtime] = val

        return results
