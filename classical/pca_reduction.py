"""
pca_reduction.py
===============
Module de réduction de dimensionnalité par Analyse en Composantes Principales (PCA)

Référence Paper: Section IV-B (Hybrid-Quantum Model Architecture)
Auteur: Basé sur l'architecture de Ridha Azizi et al.
Objectif: Réduire les features ResNet50 (2048-dim) → 8-dim pour circuit quantique

Spécifications Papier (Page 12):
"To adapt the classical features to current quantum-hardware constraints, Principal 
Component Analysis (PCA) reduces the feature vector from 2048 to 8 dimensions while 
preserving approximately 85% of the variance."

"These 8 features allow the model to fit within the limited qubit count of NISQ devices 
while retaining the most discriminative information extracted by the CNN."
"""

import numpy as np
import pickle
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader


# =============================================================================
# CONFIGURATION PCA (Basée sur le Papier)
# =============================================================================

PCA_CONFIG = {
    # Dimension cible = nombre de qubits dans le circuit quantique
    'n_components': 8,           # 8 qubits (paper: "8-dimensional classical features")
    
    # Seuil minimum de variance à préserver
    'min_variance_ratio': 0.85,  # "~85% of the variance" (exigence papier)
    
    # Standardiser avant PCA? (recommandé pour ResNet features)
    'standardize': True,
}


# =============================================================================
# CLASSE PRINCIPALE: PCAReducer
# =============================================================================

class PCAReducer:
    """
    Réducteur PCA pour adapter les features CNN aux contraintes du hardware quantique.
    
    Workflow:
    1. fit(X_train): Entraîner le PCA sur les données d'entraînement SEULEMENT
    2. transform(X): Réduire n'importe quelles données (train/val/test)
    3. inverse_transform(X): Reconstruire (approximativement) les données originales
    
    Attributs après fit():
    - pca: Modèle PCA sklearn entraîné
    - scaler: Scaler standardisé (si utilisé)
    - variance_preserved: Ratio de variance conservée
    - n_components_: Nombre effectif de composantes
    """
    
    def __init__(self, n_components=8, standardize=True, min_variance=0.85):
        """
        Initialise le réducteur PCA.
        
        Args:
            n_components (int): Nombre de dimensions cibles (doit = nombre de qubits)
            standardize (bool): Standardiser les données avant PCA (recommandé)
            min_variance (float): Variance minimale à préserver (validation)
        """
        self.n_components = n_components
        self.standardize = standardize
        self.min_variance = min_variance
        
        # Initialiser les modèles (seront entraînés dans fit())
        self.pca = None
        self.scaler = StandardScaler() if standardize else None
        
        # Métriques après fit()
        self.variance_preserved = None
        self.explained_variance_ratio_ = None
        self.is_fitted = False
        
        print(f"🔧 PCAReducer initialisé: {n_components} composantes cibles")
    
    def fit(self, X_train):
        """
        Entraîne le réducteur PCA sur les données d'entraînement.
        
        IMPORTANT: Ne doit être appelé QUE sur les données d'entraînement
        pour éviter le data leakage!
        
        Args:
            X_train (np.ndarray or torch.Tensor): Features d'entraînement
                                                  Shape: [n_samples, 2048]
        
        Returns:
            self: Instance entraînée (permet chaining)
        """
        print("\n" + "="*60)
        print("📊 ENTRAÎNEMENT PCA")
        print("="*60)
        
        # Convertir en numpy si nécessaire
        if isinstance(X_train, torch.Tensor):
            X_train = X_train.detach().cpu().numpy()
        
        X_train = np.array(X_train)
        
        print(f"📥 Données d'entrée: shape={X_train.shape}")
        print(f"   Dimensions originales: {X_train.shape[1]}")
        print(f"   Nombre d'échantillons: {X_train.shape[0]}")
        
        # Étape 1: STANDARDISATION (si activée)
        if self.standardize and self.scaler is not None:
            print("\n⚙️  Standardisation des données...")
            X_train_scaled = self.scaler.fit_transform(X_train)
            print("   ✅ Standardisation terminée (mean=0, std=1)")
        else:
            X_train_scaled = X_train.copy()
        
        # Étape 2: PCA PROPREMENT DIT
        print(f"\n🔄 Application PCA ({X_train_scaled.shape[1]} → {self.n_components})...")
        
        self.pca = PCA(n_components=self.n_components, random_state=42)
        X_reduced = self.pca.fit_transform(X_train_scaled)
        
        # Calculer les métriques de variance
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        self.variance_preserved = sum(self.explained_variance_ratio_)
        
        print(f"\n📈 RÉSULTATS PCA:")
        print(f"   Composantes principales: {self.n_components}")
        print(f"   Variance totale préservée: {self.variance_preserved:.4f} ({self.variance_preserved*100:.2f}%)")
        
        # Validation du seuil de variance
        if self.variance_preserved >= self.min_variance:
            print(f"   ✅ SEUIL ATTEINT: ≥{self.min_variance*100:.0f}% variance préservée")
        else:
            print(f"   ⚠️  ATTENTION: Variance < {self.min_variance*100:.0f}%")
            print(f"   Considérer d'augmenter n_components")
        
        # Afficher la variance par composante
        print(f"\n📊 Variance par composante:")
        for i, var in enumerate(self.explained_variance_ratio_):
            cumulative = sum(self.explained_variance_ratio_[:i+1])
            print(f"   PC{i+1}: {var*100:.2f}% (cumul: {cumulative*100:.2f}%)")
        
        # Marquer comme entraîné
        self.is_fitted = True
        
        print(f"\n✅ PCA entraîné avec succès!")
        print(f"   Shape sortie: {X_reduced.shape}")
        
        return self
    
    def transform(self, X):
        """
        Applique la réduction PCA à de nouvelles données.
        
        Utilise les paramètres appris lors de fit() (moyenne, écart-type, composantes).
        
        Args:
            X (np.ndarray or torch.Tensor): Données à réduire
                                           Shape: [n_samples, 2048]
        
        Returns:
            np.ndarray: Données réduites Shape: [n_samples, 8]
        
        Raises:
            ValueError: Si PCA n'a pas été entraîné (fit() non appelé)
        """
        if not self.is_fitted:
            raise ValueError("❌ ERREUR: PCA pas encore entraîné! Appeler fit() d'abord.")
        
        # Convertir en numpy
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        
        X = np.array(X)
        
        # Standardiser (avec paramètres d'entraînement)
        if self.standardize and self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # Appliquer PCA
        X_reduced = self.pca.transform(X_scaled)
        
        return X_reduced
    
    def inverse_transform(self, X_reduced):
        """
        Reconstruit approximativement les données originales depuis l'espace réduit.
        
        Utile pour visualisation ou analyse d'erreur de reconstruction.
        
        Args:
            X_reduced (np.ndarray): Données réduites Shape: [n_samples, 8]
        
        Returns:
            np.ndarray: Données reconstruites Shape: [n_samples, 2048]
        """
        if not self.is_fitted:
            raise ValueError("❌ ERREUR: PCA pas encore entraîné!")
        
        # Inverse PCA
        X_scaled = self.pca.inverse_transform(X_reduced)
        
        # Inverse standardisation
        if self.standardize and self.scaler is not None:
            X_original = self.scaler.inverse_transform(X_scaled)
        else:
            X_original = X_scaled
        
        return X_original
    
    def get_feature_importance(self, original_dim_names=None):
        """
        Calcule l'importance de chaque feature originale dans les composantes PCA.
        
        Utile pour interpréter quelles features ResNet50 contribuent le plus.
        
        Args:
            original_dim_names (list): Noms des features originales (optionnel)
        
        Returns:
            dict: Importances par composante principale
        """
        if not self.is_fitted:
            raise ValueError("❌ ERREUR: PCA pas encore entraîné!")
        
        importance_dict = {}
        
        for i in range(self.n_components):
            # Prendre la valeur absolue des loadings
            loadings = abs(self.pca.components_[i])
            
            # Top 10 features les plus importantes pour cette composante
            top_indices = np.argsort(loadings)[::-1][:10]
            
            component_info = {
                'variance_explained': self.explained_variance_ratio_[i],
                'top_features_indices': top_indices.tolist(),
                'top_features_loadings': loadings[top_indices].tolist()
            }
            
            if original_dim_names:
                component_info['top_features_names'] = [
                    original_dim_names[idx] for idx in top_indices
                ]
            
            importance_dict[f'PC{i+1}'] = component_info
        
        return importance_dict
    
    def visualize_variance(self, save_path=None):
        """
        Visualise la variance expliquée par chaque composante.
        
        Génère deux graphiques:
        1. Barplot de variance individuelle par composante
        2. Courbe de variance cumulée
        
        Args:
            save_path (str): Chemin pour sauvegarder (optionnel)
        """
        if not self.is_fitted:
            raise ValueError("❌ ERREUR: PCA pas encore entraîné!")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Graphique 1: Variance individuelle
        x_pos = range(1, self.n_components + 1)
        bars = ax1.bar(x_pos, self.explained_variance_ratio_ * 100, 
                       color='#1a5490', alpha=0.7, edgecolor='black')
        
        ax1.set_xlabel('Composante Principale', fontsize=11)
        ax1.set_ylabel('Variance Expliquée (%)', fontsize=11)
        ax1.set_title('Variance par Composante Principale', fontsize=12, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.axhline(y=100/self.n_components, color='r', linestyle='--', 
                    label=f'Moyenne ({100/self.n_components:.1f}%)')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Ajouter valeurs sur les barres
        for bar, val in zip(bars, self.explained_variance_ratio_ * 100):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # Graphique 2: Variance cumulée
        cumulative_variance = np.cumsum(self.explained_variance_ratio_) * 100
        ax2.plot(x_pos, cumulative_variance, 'o-', color='#28a745', linewidth=2, 
                 markersize=8, label='Variance Cumulée')
        ax2.fill_between(x_pos, cumulative_variance, alpha=0.3, color='#28a745')
        
        # Ligne seuil 85%
        ax2.axhline(y=self.min_variance * 100, color='red', linestyle='--', 
                    linewidth=2, label=f'Seuil {self.min_variance*100:.0f}%')
        
        ax2.set_xlabel('Nombre de Composantes', fontsize=11)
        ax2.set_ylabel('Variance Cumulée (%)', fontsize=11)
        ax2.set_title('Variance Cumulée', fontsize=12, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_ylim([0, 105])
        ax2.legend(loc='lower right')
        ax2.grid(alpha=0.3)
        
        # Annoter le point final
        ax2.annotate(f'{cumulative_variance[-1]:.1f}%', 
                     xy=(self.n_components, cumulative_variance[-1]),
                     xytext=(self.n_components-1.5, cumulative_variance[-1]+5),
                     fontsize=10, fontweight='bold', color='#28a745',
                     arrowprops=dict(arrowstyle='->', color='#28a745'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Graphique PCA sauvegardé: {save_path}")
        
        plt.show()
    
    def save_model(self, filepath):
        """
        Sauvegarde le modèle PCA entraîné sur disque.
        
        Args:
            filepath (str): Chemin de sauvegarde (.pkl)
        """
        if not self.is_fitted:
            raise ValueError("❌ ERREUR: Rien à sauvegarder - PCA pas entraîné!")
        
        model_data = {
            'pca': self.pca,
            'scaler': self.scaler,
            'n_components': self.n_components,
            'variance_preserved': self.variance_preserved,
            'explained_variance_ratio': self.explained_variance_ratio_,
            'is_fitted': self.is_fitted,
            'config': PCA_CONFIG
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Modèle PCA sauvegardé: {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """
        Charge un modèle PCA sauvegardé.
        
        Args:
            filepath (str): Chemin du fichier .pkl
        
        Returns:
            PCAReducer: Instance chargée et prête à transformer
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls(
            n_components=model_data['n_components'],
            standardize=model_data['scaler'] is not None
        )
        
        instance.pca = model_data['pca']
        instance.scaler = model_data['scaler']
        instance.variance_preserved = model_data['variance_preserved']
        instance.explained_variance_ratio_ = model_data['explained_variance_ratio']
        instance.is_fitted = model_data['is_fitted']
        
        print(f"📂 Modèle PCA chargé: {filepath}")
        print(f"   Variance préservée: {instance.variance_preserved*100:.2f}%")
        
        return instance


# =============================================================================
# FONCTIONS UTILITAIRES INTÉGRATION PIPELINE
# =============================================================================

def extract_and_reduce_features(model, dataloader, pca_reducer, device='cpu'):
    """
    Extrait les features ResNet50 et applique la réduction PCA en une seule étape.
    
    Combine l'extraction CNN et la réduction de dimensionnalité pour le pipeline hybride.
    
    Args:
        model (nn.Module): Modèle ResNet50 extractor (en mode eval)
        dataloader: DataLoader PyTorch contenant les images
        pca_reducer (PCAReducer): PCA déjà entraîné
        device (str): 'cpu' ou 'cuda'
    
    Returns:
        tuple: (features_reduced, labels_true)
               - features_reduced: np.ndarray shape [n_samples, 8]
               - labels_true: np.ndarray shape [n_samples]
    """
    model.eval()
    model.to(device)
    
    all_features_raw = []
    all_labels = []
    
    print(f"\n🔄 Extraction & Réduction PCA sur {len(dataloader)} batches...")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            
            # Extraire features ResNet50 (2048-dim)
            features_raw = model(images)  # [batch, 2048]
            
            all_features_raw.append(features_raw.cpu().numpy())
            all_labels.append(labels.numpy())
            
            if (batch_idx + 1) % 50 == 0:
                print(f"   Batch {batch_idx + 1}/{len(dataloader)} traité")
    
    # Concaténer tous les batches
    all_features_raw = np.vstack(all_features_raw)
    all_labels = np.concatenate(all_labels)
    
    print(f"   Features brutes extraites: {all_features_raw.shape}")
    
    # Appliquer PCA (2048 → 8)
    features_reduced = pca_reducer.transform(all_features_raw)
    
    print(f"   Features réduites (PCA): {features_reduced.shape}")
    print(f"   ✅ Extraction & réduction terminées!")
    
    return features_reduced, all_labels


# =============================================================================
# POINT D'ENTRÉE PRINCIPAL (TEST)
# =============================================================================

if __name__ == "__main__":
    """
    Script de test pour vérifier le module PCA.
    
    Génère des données synthétiques simulant des features ResNet50 pour tester.
    """
    
    print("=" * 70)
    print("🧪 TEST DU MODULE PCA RÉDUCTION")
    print("=" * 70)
    
    # Générer des données synthétiques (simulant 1000 images, 2048 features ResNet50)
    np.random.seed(42)
    n_samples = 1000
    n_features = 2048  # ResNet50 output dimension
    
    print(f"\n📊 Génération de données synthétiques...")
    print(f"   Samples: {n_samples}, Features: {n_features}")
    
    # Simuler des features avec structure (corrélations entre groupes de features)
    X_synthetic = np.random.randn(n_samples, n_features)
    
    # Ajouter de la structure (corrélations locales)
    for i in range(0, n_features, 64):  # Groupes de 64 features corrélées
        base_pattern = np.random.randn(n_samples)
        for j in range(i, min(i + 64, n_features)):
            X_synthetic[:, j] += 0.5 * base_pattern + np.random.randn(n_samples) * 0.1
    
    # Split synthétique train/test (80/20)
    split_idx = int(0.8 * n_samples)
    X_train = X_synthetic[:split_idx]
    X_test = X_synthetic[split_idx:]
    
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Tester le PCA
    try:
        # Initialiser
        pca = PCAReducer(n_components=8, standardize=True, min_variance=0.85)
        
        # Entraîner sur training set
        pca.fit(X_train)
        
        # Transformer les deux ensembles
        X_train_reduced = pca.transform(X_train)
        X_test_reduced = pca.transform(X_test)
        
        print(f"\n📐 Résultats Transformation:")
        print(f"   Train réduit: {X_train_reduced.shape}")
        print(f"   Test réduit:  {X_test_reduced.shape}")
        
        # Tester inverse transform
        X_reconstructed = pca.inverse_transform(X_train_reduced)
        reconstruction_error = np.mean((X_train - X_reconstructed) ** 2)
        print(f"\n📉 Erreur de reconstruction (MSE): {reconstruction_error:.6f}")
        
        # Visualiser
        print("\n📊 Génération des graphiques de variance...")
        pca.visualize_variance(save_path='pca_variance_analysis.png')
        
        # Sauvegarder/Charger le modèle
        print("\n💾 Test sauvegarde/chargement...")
        pca.save_model('test_pca_model.pkl')
        pca_loaded = PCAReducer.load_model('test_pca_model.pkl')
        
        # Vérifier que le chargé donne mêmes résultats
        X_test_loaded = pca_loaded.transform(X_test)
        assert np.allclose(X_test_reduced, X_test_loaded), "❌ Chargement incorrect!"
        print("   ✅ Sauvegarde/chargement vérifié!")
        
        # Nettoyer fichier test
        if os.path.exists('test_pca_model.pkl'):
            os.remove('test_pca_model.pkl')
        if os.path.exists('pca_variance_analysis.png'):
            os.remove('pca_variance_analysis.png')
        
        print("\n" + "="*70)
        print("🎉 SUCCÈS! Module pca_reduction.py fonctionne correctement!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()