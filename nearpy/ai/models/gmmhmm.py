from hmmlearn import hmm
from sklearn.mixture import GaussianMixture
import numpy as np
import os
import pickle

class GMMHMMVowelClassifier:
    def __init__(self, n_states=5, n_mix=3, cov_type='diag', n_iter=10):
        self.n_states = n_states
        self.n_mix = n_mix
        self.cov_type = cov_type
        self.n_iter = n_iter
        self.models = {}
        
    def train(self, features_dict):
        """
        Train GMM-HMM for each vowel class
        
        Parameters:
        features_dict: Dictionary with vowel labels as keys and list of feature arrays as values
        """
        for vowel, feature_list in features_dict.items():
            # Initialize model for this vowel
            model = hmm.GMMHMM(n_components=self.n_states, 
                              n_mix=self.n_mix,
                              covariance_type=self.cov_type, 
                              n_iter=self.n_iter)
            
            # Prepare training data
            # Concatenate all feature arrays for this vowel
            X = np.vstack(feature_list)
            lengths = [x.shape[0] for x in feature_list]
            
            # Train the model
            model.fit(X, lengths=lengths)
            
            # Save the trained model
            self.models[vowel] = model
            
    def predict(self, features):
        """
        Predict vowel class for given features
        
        Parameters:
        features: Feature array for a single audio snippet
        
        Returns:
        predicted_vowel: Predicted vowel label
        """
        if len(self.models) == 0:
            raise ValueError("Models not trained yet!")
        
        # Calculate log likelihood for each model
        log_likelihoods = {}
        for vowel, model in self.models.items():
            log_likelihoods[vowel] = model.score(features)
        
        # Return the vowel with highest log likelihood
        predicted_vowel = max(log_likelihoods, key=log_likelihoods.get)
        return predicted_vowel
    
    def save_models(self, directory):
        """Save trained models to disk"""
        os.makedirs(directory, exist_ok=True)
        for vowel, model in self.models.items():
            with open(f"{directory}/{vowel}_model.pkl", 'wb') as f:
                pickle.dump(model, f)
                
    def load_models(self, directory):
        """Load trained models from disk"""
        for filename in os.listdir(directory):
            if filename.endswith("_model.pkl"):
                vowel = filename.split("_model.pkl")[0]
                with open(f"{directory}/{filename}", 'rb') as f:
                    self.models[vowel] = pickle.load(f)