import pickle

import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils._tags import TargetTags
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from scikit_rca.utils.data import PairDataset
from scikit_rca.utils.metrics import contrastive_loss, icc11, info_nce


class RCA(TransformerMixin, BaseEstimator):
    """Reliability Component Analysis transformer.

    Linear: Starting with the feature matrix Z^0 of size [num_samples,
    num_features], we fit a matrix W^0 of size [num_features, 1], giving a
    Y^0 = Z^0 W^0 of size [N,1]. Then we project each row of Z^0 onto W^0,
    and subtract this projection from Z^0, giving a new feature matrix Z^1 of
    size [num_samples, num_features]. We repeat this process until we have
    extracted n_components components Y^0,...,Y^{n_components-1} of size
    [num_samples, 1].

    Nonlinear: Use a neural network to project the data onto a lower-dimensional
    space. The network is trained to minimize the contrastive loss between pairs
    of samples that belong to the same subject, and pairs that belong to
    different subjects. The output of the network is then used as the embedding
    for each sample. The output of the network is Y of size
    [num_samples, n_components].

    Parameters
    ----------
    n_components : int
        Number of components to extract.
    model_type : {"linear", "linear_with_multicomponent", "nonlinear"},
        default="linear"
        Model selection.
    eps : float, default=1.0
        Epsilon parameter for the contrastive loss.
    lr : float, default=1e-4
        Learning rate for the optimizer.
    n_epochs : int, default=10
        Number of training epochs.
    batch_size : int, default=10
        Batch size for training.
    weight_decay : float, default=1e-8
        Weight decay parameter for the optimizer.
    device : str, default="cpu"
        Torch device string.
    loss_type : {"contrastive", "info_nce"}, default="contrastive"
        Loss function type.
    verbose : bool, default=False
        Whether to print progress messages during fitting.
    orthogonality_penalty : {"participants", "weights"}, default="participants"
        Orthogonality penalty type.
    orthogonality_by_correlation : bool, default=True
        Use correlation instead of dot product for orthogonality.
    penalty_scale : float or None, default=None
        Multiplier for orthogonality penalty.
    """

    _parameter_constraints = {}

    def __init__(
        self,
        n_components=1,
        model_type="linear",
        eps=1.0,
        lr=1e-4,
        n_epochs=10,
        batch_size=10,
        weight_decay=1e-8,
        device="cpu",
        loss_type="contrastive",
        random_state=None,
        verbose=False,
        orthogonality_penalty="participants",
        orthogonality_by_correlation=True,
        penalty_scale=None,
    ):
        self.n_components = n_components
        self.device = device
        self.model_type = model_type
        self.loss_type = loss_type
        self.random_state = random_state
        self.verbose = verbose
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.eps = eps
        self.orthogonality_penalty = orthogonality_penalty
        self.orthogonality_by_correlation = orthogonality_by_correlation
        self.penalty_scale = penalty_scale

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags = TargetTags(required=True)
        return tags

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """
        X: shape [num_samples, d]
        y: Identifies pairs of indices of which samples belong to the same
           subject. Shape [num_samples, 2]. Each entry must be an int between
           0 and labels.shape[0]-1
        """
        X = validate_data(self, X, accept_sparse=False)
        self.n_features_ = X.shape[1]
        labels = self._coerce_labels(X, y)
        self.losses_ = []
        self.weights_ = []  # in the linear case each weight has shape [1, num_features]
        X = X.astype("float32")
        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)
        self._check_dimensions(X, labels)
        X, labels = self._convert_to_torch(X, labels)
        for k in range(0, self.n_components):
            if self.verbose:
                print(f"Fitting component {k+1}")
            if self.model_type == "linear":
                weights, losses = self._fit_component(X, labels)
            elif self.model_type == "nonlinear" or self.model_type == "linear_with_multicomponent":
                if k == 0:
                    weights, losses = self._fit_component(X, labels)
                else:
                    break
            self.losses_.append(losses)
            self.weights_.append(weights)
        self.losses_ = np.asarray(self.losses_)
        if isinstance(self.weights_[-1], np.ndarray):
            weights = np.asarray(self.weights_)
            if weights.ndim == 3 and weights.shape[1] == 1:
                weights = weights.squeeze(1)
            self.weights_ = weights
        self.is_fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self, ["weights_"])
        X = validate_data(self, X, accept_sparse=False, reset=False)
        embeddings = []
        output_dtype = X.dtype
        X_float = X.astype("float32")
        if isinstance(self.weights_, np.ndarray) and self.weights_.ndim == 1:
            weights = self.weights_[None, :]
        else:
            weights = self.weights_
        for i in range(0, len(weights)):
            if self.model_type == "linear":
                embeddings.append((X_float @ weights[i][:, None])[:, 0])
            elif self.model_type == "nonlinear" or self.model_type == "linear_with_multicomponent":
                # Nonlinear/multicomponent uses a single model to produce all components.
                curr_embed = weights[i](torch.from_numpy(X_float)).detach().numpy()
                embeddings = curr_embed.T
        embeddings = np.asarray(embeddings).T
        if len(embeddings.shape) < 2:
            embeddings = np.expand_dims(embeddings, axis=-1)
        return embeddings.astype(output_dtype, copy=False)

    def score(self, X, y=None, dim=0):
        check_is_fitted(self, ["weights_"])
        labels = self._coerce_labels(X, y)
        embedding = self.transform(X)[:, dim]
        return icc11(labels[:, 0], embedding, return_stats=False)

    def orthogonality_check(self):
        for i in range(1, len(self.weights_)):
            for j in range(0, i):
                if self.model_type == "linear":
                    if self.verbose:
                        print(
                            "Orthogonality between component "
                            f"{i} and {j}: {np.abs(self.weights_[i] @ self.weights_[j].T)}"
                        )

    def _fit_component(self, X, labels):
        # Create data loader for training
        dataset = PairDataset(X, labels, device=self.device)
        generator = None
        if self.random_state is not None:
            generator = torch.Generator()
            generator.manual_seed(self.random_state)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )
        if self.model_type == "linear":
            model = nn.Linear(self.n_features_, 1, bias=False)
        elif self.model_type == "linear_with_multicomponent":
            model = nn.Linear(self.n_features_, self.n_components, bias=False)
        elif self.model_type == "nonlinear":
            model = nn.Sequential(
                nn.Linear(self.n_features_, 10),
                nn.ReLU(),
                nn.Linear(10, self.n_components, bias=False),
            )
        else:
            raise ValueError("Model type not recognised, using linear model")
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        losses = []

        # Training loop
        pbar = tqdm(range(self.n_epochs))
        for _ in pbar:
            avg_loss = []  # average loss across batches
            avg_gn = []  # average gradient norm across batches
            # Iterate over all data to update parameters
            for batch_idx, data_X, data_labels in dataloader:
                output = model.forward(data_X)
                if self.loss_type == "contrastive":
                    loss = contrastive_loss(output, data_labels, self.eps)
                elif self.loss_type == "info_nce":
                    loss = info_nce(output, data_labels)
                loss += self._orthogonality_penalty_loss(output, data_X, model)
                optimizer.zero_grad()
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
                optimizer.step()
                # Remember loss and gradient norm per batch
                avg_loss.append(loss.detach().item())
                avg_gn.append(gn.detach().item())

            description = (
                f"Loss={np.array(avg_loss).mean():.5f} | "
                f"grad_norm={np.array(avg_gn).mean():.2f} | "
                f"lr={optimizer.param_groups[0]['lr']:.1e}"
            )
            pbar.set_description(description)

            # Logging
            losses.append(np.array(avg_loss).mean())

        if self.model_type == "linear":
            matrix = model.weight.detach().cpu().numpy()
            return matrix, losses
        return model, losses

    def _weight_penalty_scale(self):
        """Return the penalty scale for weight orthogonality."""
        return 10 if self.penalty_scale is None else self.penalty_scale

    def _participant_penalty_scale(self, z_curr):
        """Return the penalty scale for participant orthogonality."""
        base = 0.1 if self.penalty_scale is None else self.penalty_scale
        return base / z_curr.size(0) / (len(self.weights_) + 1)

    def _orthogonality_penalty_loss(self, output, data_X, model):
        if self.orthogonality_penalty == "weights":
            return self._weight_orthogonality_loss(model)
        if self.orthogonality_penalty == "participants":
            return self._participant_orthogonality_loss(output, data_X)
        return 0

    def _weight_orthogonality_loss(self, model):
        penalty_scale = self._weight_penalty_scale()
        w_current = model.weight.view(-1)  # shape: (num_features,)
        loss = 0
        for w_prev in self.weights_:
            w_prev = torch.tensor(w_prev).view(-1).to(w_current.device)  # shape: (num_features,)
            if self.orthogonality_by_correlation:
                dot = torch.dot(w_current - w_current.mean(), w_prev - w_prev.mean())
            else:
                dot = torch.dot(w_current, w_prev)
            loss += penalty_scale * torch.pow(dot, 2)
        return loss

    def _participant_orthogonality_loss(self, output, data_X):
        z_curr = output.view(-1)  # (batch,)
        penalty_scale = self._participant_penalty_scale(z_curr)
        loss = 0
        for w_prev_np in self.weights_:  # loop over old comps
            w_prev = torch.as_tensor(
                w_prev_np,
                device=z_curr.device,
                dtype=z_curr.dtype,
            ).view(
                -1
            )  # (n_features,)
            z_prev = data_X @ w_prev  # (batch,)
            if self.orthogonality_by_correlation:
                dot = torch.dot(z_curr - z_curr.mean(), z_prev - z_prev.mean())
                loss += penalty_scale * dot.pow(2)
            else:
                dot = torch.dot(z_curr, z_prev)
                loss += (penalty_scale / 10) * dot.pow(2)
        return loss

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        return True

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def _check_dimensions(X, labels):
        labels = np.asarray(labels)
        if not X.ndim == 2:
            raise ValueError(f"Input X must be of shape [num_scans, d]. Got {X.shape}")
        if not labels.ndim == 2 or not labels.shape[1] == 2:
            raise ValueError(f"Input y must be of shape [num_scans, 2]. Got {labels.shape}")
        if not X.shape[0] == labels.shape[0]:
            raise ValueError(
                f"Input X, y must have matching first dimensions. Got X.shape={X.shape}, y.shape={labels.shape}"
            )

    @staticmethod
    def _convert_to_torch(X, labels):
        return torch.from_numpy(X), torch.from_numpy(labels)

    @staticmethod
    def _coerce_labels(X, labels):
        if labels is None:
            raise ValueError("requires y to be passed, but the target y is None")
        labels_arr = np.asarray(labels)
        if labels_arr.dtype == object:
            target_type = type_of_target(labels_arr)
            if target_type == "unknown":
                raise ValueError("Unknown label type: object")
        if labels_arr.ndim == 1:
            if labels_arr.shape[0] != X.shape[0]:
                raise ValueError(
                    "Input X, labels must have matching first dimensions. "
                    f"Got X.shape={X.shape}, labels.shape={labels_arr.shape}"
                )
            labels_arr = np.column_stack((labels_arr, np.arange(labels_arr.shape[0])))
        return labels_arr
