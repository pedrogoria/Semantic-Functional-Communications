"""
sfc/core/fading.py

Trusted and extensible core module for fading generation.

This module is designed to support:
- controlled random generation through an object-level seed;
- multiple fading models in a single consistent API;
- backward compatibility with legacy function-style usage.

Currently implemented models:
- Rice / Rician fading
- Rayleigh fading
- Nakagami-m fading
- Weibull fading
- Deterministic / no-fading channel

The design also supports:
- future custom fading-model registration
- future extension without changing the public API

IMPORTANT
---------
This file is part of the scientific core.

The current Rice fading implementation preserves the mathematical model
used in the original trusted code:
    h = (sigma * N(0,1) + mu) + j * (sigma * N(0,1) + mu)
    return abs(h)

The new class-based structure extends the module but does not invalidate
the original model.
"""

import numpy as np


class FadingGenerator:
    """
    Generate fading samples using a reproducible object-level random generator.

    Why a class is appropriate here
    -------------------------------
    Unlike a purely stateless utility function, fading generation often benefits
    from object-level control over randomness. In research software, this helps
    with:
    - reproducibility;
    - experiment isolation;
    - deterministic debugging;
    - future extension to correlated fading, time-evolution, and custom models.

    Parameters
    ----------
    seed : int or None, optional
        Seed used to initialize the internal random number generator.
        If None, NumPy will initialize the generator in a non-deterministic way.

    rng : numpy.random.Generator or numpy.random.RandomState or None, optional
        External random-number generator. If provided, it is used directly and
        the `seed` argument is ignored.

    use_legacy_random_state : bool, optional
        If True and `rng` is not provided, the internal RNG will be
        `np.random.RandomState(seed)`, which is closer to older NumPy behavior.
        If False, the internal RNG will be `np.random.default_rng(seed)`,
        which is the modern NumPy API.

    Notes
    -----
    The object abstracts away the random backend by exposing private helper
    methods (`_standard_normal`, `_uniform`, `_gamma`, `_weibull`).
    This keeps the fading formulas independent of the RNG implementation.
    """

    def __init__(self, seed=None, rng=None, use_legacy_random_state=False):
        """
        Initialize the fading generator.
        """

        if rng is not None:
            self.rng = rng
        else:
            if use_legacy_random_state:
                self.rng = np.random.RandomState(seed)
            else:
                self.rng = np.random.default_rng(seed)

        # ------------------------------------------------------------------
        # Registry for custom and built-in models.
        #
        # Each entry maps a public model name to a bound method that generates
        # fading samples. This design allows future extension without changing
        # the public generate(...) API.
        # ------------------------------------------------------------------
        self._registry = {
            "rice": self.rice,
            "rician": self.rice,
            "ricean": self.rice,      # kept for legacy naming compatibility
            "rayleigh": self.rayleigh,
            "nakagami": self.nakagami,
            "nakagami-m": self.nakagami,
            "weibull": self.weibull,
            "none": self.no_fading,
            "deterministic": self.no_fading,
        }

    # ======================================================================
    # RNG helper methods
    # ======================================================================

    def _standard_normal(self, size):
        """
        Draw standard normal samples using the configured RNG backend.

        Parameters
        ----------
        size : int or tuple
            Output shape.

        Returns
        -------
        np.ndarray
            Standard normal samples.
        """

        # NumPy Generator and RandomState both provide .standard_normal(...)
        return self.rng.standard_normal(size)

    def _uniform(self, low=0.0, high=1.0, size=None):
        """
        Draw uniform samples using the configured RNG backend.

        Parameters
        ----------
        low : float
            Lower bound.

        high : float
            Upper bound.

        size : int or tuple or None
            Output shape.

        Returns
        -------
        np.ndarray
            Uniform random samples.
        """

        # Both Generator and RandomState expose .uniform(...)
        return self.rng.uniform(low=low, high=high, size=size)

    def _gamma(self, shape, scale=1.0, size=None):
        """
        Draw gamma-distributed samples using the configured RNG backend.

        Parameters
        ----------
        shape : float
            Shape parameter of the Gamma distribution.

        scale : float
            Scale parameter of the Gamma distribution.

        size : int or tuple or None
            Output shape.

        Returns
        -------
        np.ndarray
            Gamma random samples.
        """

        # Both Generator and RandomState expose .gamma(...)
        return self.rng.gamma(shape=shape, scale=scale, size=size)

    def _weibull(self, a, size=None):
        """
        Draw Weibull-distributed samples using the configured RNG backend.

        Parameters
        ----------
        a : float
            Weibull shape parameter.

        size : int or tuple or None
            Output shape.

        Returns
        -------
        np.ndarray
            Weibull random samples.
        """

        # Both Generator and RandomState expose .weibull(...)
        return self.rng.weibull(a=a, size=size)

    # ======================================================================
    # Public registry helpers
    # ======================================================================

    def available_models(self):
        """
        Return the sorted list of available model names.

        Returns
        -------
        list of str
            Registered fading-model names.
        """

        return sorted(self._registry.keys())

    def register_model(self, name, generator_function):
        """
        Register a custom fading model.

        Parameters
        ----------
        name : str
            Public model name.

        generator_function : callable
            Callable with signature compatible with:
                generator_function(n=1, return_complex=False, **kwargs)

        Notes
        -----
        The function is stored directly in the registry. This allows external
        research extensions without touching the core file.
        """

        self._registry[name.lower()] = generator_function

    def generate(self, model="rice", n=1, return_complex=False, **kwargs):
        """
        Generate fading samples using a model name.

        Parameters
        ----------
        model : str
            Name of the fading model.

        n : int, optional
            Number of samples.

        return_complex : bool, optional
            If True, return complex coefficients when the model supports it.
            If False, return the fading envelope magnitude.

        **kwargs
            Extra model-specific parameters.

        Returns
        -------
        np.ndarray
            Generated fading samples.

        Raises
        ------
        ValueError
            If the requested model is not registered.
        """

        model_key = model.lower()

        if model_key not in self._registry:
            raise ValueError(
                f"Unknown fading model '{model}'. "
                f"Available models: {self.available_models()}"
            )

        return self._registry[model_key](
            n=n,
            return_complex=return_complex,
            **kwargs
        )

    # ======================================================================
    # Built-in fading models
    # ======================================================================

    def no_fading(self, n=1, return_complex=False, amplitude=1.0, **kwargs):
        """
        Generate a deterministic channel with no fading.

        Parameters
        ----------
        n : int, optional
            Number of samples.

        return_complex : bool, optional
            If True, return a complex constant.
            If False, return a real constant.

        amplitude : float, optional
            Constant amplitude of the channel.

        Returns
        -------
        np.ndarray
            Deterministic channel samples.
        """

        if return_complex:
            return amplitude * np.ones(n, dtype=complex)

        return amplitude * np.ones(n)

    def rice(self, K_dB=1, n=1, return_complex=False, **kwargs):
        """
        Generate Rice / Rician fading samples.

        This method preserves the same mathematical model used in the original
        trusted code for the envelope:

            k = 10 ** (K_dB / 10)
            mu = sqrt(k / (2 * (k + 1)))
            sigma = sqrt(1 / (2 * (k + 1)))
            h = (sigma*N1 + mu) + j*(sigma*N2 + mu)
            return abs(h)

        Parameters
        ----------
        K_dB : float, optional
            Ricean K-factor in dB.

        n : int, optional
            Number of samples.

        return_complex : bool, optional
            If True, return the complex fading coefficient h.
            If False, return the envelope abs(h), which matches the original
            function behavior.

        Returns
        -------
        np.ndarray
            Complex Rice fading coefficients or their envelope magnitudes.
        """

        # ------------------------------------------------------------------
        # Convert K-factor from dB scale to linear scale.
        # ------------------------------------------------------------------
        k = 10 ** (K_dB / 10)

        # ------------------------------------------------------------------
        # Mean of the in-phase and quadrature Gaussian components.
        # This preserves the same original parameterization.
        # ------------------------------------------------------------------
        mu = np.sqrt(k / (2 * (k + 1)))

        # ------------------------------------------------------------------
        # Standard deviation of the Gaussian components.
        # This preserves the same original parameterization.
        # ------------------------------------------------------------------
        sigma = np.sqrt(1 / (2 * (k + 1)))

        # ------------------------------------------------------------------
        # Generate the complex fading coefficient exactly in the same form as
        # the original trusted implementation.
        # ------------------------------------------------------------------
        h = (
            sigma * self._standard_normal(n) + mu
        ) + 1j * (
            sigma * self._standard_normal(n) + mu
        )

        if return_complex:
            return h

        # ------------------------------------------------------------------
        # Preserve the envelope-only behavior of the original function.
        # ------------------------------------------------------------------
        return np.abs(h)

    def rayleigh(self, n=1, sigma=None, return_complex=False, **kwargs):
        """
        Generate Rayleigh fading samples.

        Parameters
        ----------
        n : int, optional
            Number of samples.

        sigma : float or None, optional
            Standard deviation of the real and imaginary Gaussian components.
            If None, the default is 1/sqrt(2), which yields unit average power
            for the complex coefficient.

        return_complex : bool, optional
            If True, return the complex fading coefficient.
            If False, return the envelope magnitude.

        Returns
        -------
        np.ndarray
            Complex Rayleigh fading coefficients or their envelope magnitudes.

        Notes
        -----
        Rayleigh fading is obtained as a zero-mean circularly symmetric complex
        Gaussian coefficient:
            h = sigma*N1 + j*sigma*N2
        """

        if sigma is None:
            sigma = 1 / np.sqrt(2)

        h = (
            sigma * self._standard_normal(n)
        ) + 1j * (
            sigma * self._standard_normal(n)
        )

        if return_complex:
            return h

        return np.abs(h)

    def nakagami(self, m=1.0, omega=1.0, n=1, return_complex=False, **kwargs):
        """
        Generate Nakagami-m fading samples.

        Parameters
        ----------
        m : float, optional
            Nakagami shape parameter.
            Must satisfy m > 0.

        omega : float, optional
            Average power parameter.
            Must satisfy omega > 0.

        n : int, optional
            Number of samples.

        return_complex : bool, optional
            If True, generate a complex coefficient whose envelope follows the
            Nakagami-m distribution and whose phase is uniform in [0, 2*pi).
            If False, return only the envelope.

        Returns
        -------
        np.ndarray
            Nakagami envelope samples or complex coefficients.

        Raises
        ------
        ValueError
            If m <= 0 or omega <= 0.

        Notes
        -----
        If R is Nakagami-m distributed, then:
            R^2 ~ Gamma(shape=m, scale=omega/m)

        Therefore, this implementation generates:
            power ~ Gamma(m, omega/m)
            envelope = sqrt(power)

        For the complex version, a uniform random phase is added:
            h = envelope * exp(j*phi)
        """

        if m <= 0:
            raise ValueError("Nakagami parameter 'm' must be strictly positive.")

        if omega <= 0:
            raise ValueError("Nakagami parameter 'omega' must be strictly positive.")

        power = self._gamma(shape=m, scale=omega / m, size=n)
        envelope = np.sqrt(power)

        if return_complex:
            phase = self._uniform(low=0.0, high=2 * np.pi, size=n)
            return envelope * np.exp(1j * phase)

        return envelope

    def weibull(self, beta=2.0, omega=1.0, n=1, return_complex=False, **kwargs):
        """
        Generate Weibull fading samples.

        Parameters
        ----------
        beta : float, optional
            Weibull shape parameter.
            Must satisfy beta > 0.

        omega : float, optional
            Scale-related power parameter.
            Must satisfy omega > 0.

        n : int, optional
            Number of samples.

        return_complex : bool, optional
            If True, return a complex coefficient with Weibull-distributed
            envelope and uniform random phase.
            If False, return only the envelope.

        Returns
        -------
        np.ndarray
            Weibull envelope samples or complex coefficients.

        Raises
        ------
        ValueError
            If beta <= 0 or omega <= 0.

        Notes
        -----
        NumPy's weibull(a) generates samples with unit scale. Here we rescale
        them using omega to provide a convenient physical parameter.

        This is included as an extensible additional model beyond the original
        trusted implementation.
        """

        if beta <= 0:
            raise ValueError("Weibull parameter 'beta' must be strictly positive.")

        if omega <= 0:
            raise ValueError("Weibull parameter 'omega' must be strictly positive.")

        # ------------------------------------------------------------------
        # Generate unit-scale Weibull samples, then rescale by omega.
        # The rescaling is chosen so that omega acts as a scale/power control.
        # ------------------------------------------------------------------
        envelope = (omega ** 0.5) * self._weibull(a=beta, size=n)

        if return_complex:
            phase = self._uniform(low=0.0, high=2 * np.pi, size=n)
            return envelope * np.exp(1j * phase)

        return envelope


# ==========================================================================
# Backward-compatible module-level helper functions
# ==========================================================================

def ricean_fading(K_dB=1, n=1):
    """
    Backward-compatible legacy wrapper.

    This function preserves the old function-style API while delegating the
    implementation to the new class-based generator.

    Parameters
    ----------
    K_dB : float, optional
        Ricean K-factor in dB.

    n : int, optional
        Number of samples.

    Returns
    -------
    np.ndarray
        Rice fading envelope samples.
    """

    return FadingGenerator().rice(K_dB=K_dB, n=n, return_complex=False)


def rician_fading(K_dB=1, n=1):
    """
    Alias for Rice fading using the more common spelling 'Rician'.
    """

    return FadingGenerator().rice(K_dB=K_dB, n=n, return_complex=False)


def rayleigh_fading(n=1, sigma=None):
    """
    Convenience wrapper for Rayleigh fading.

    Parameters
    ----------
    n : int, optional
        Number of samples.

    sigma : float or None, optional
        Standard deviation of the Gaussian quadrature components.

    Returns
    -------
    np.ndarray
        Rayleigh fading envelope samples.
    """

    return FadingGenerator().rayleigh(n=n, sigma=sigma, return_complex=False)


def nakagami_fading(m=1.0, omega=1.0, n=1):
    """
    Convenience wrapper for Nakagami-m fading.

    Returns
    -------
    np.ndarray
        Nakagami fading envelope samples.
    """

    return FadingGenerator().nakagami(m=m, omega=omega, n=n, return_complex=False)


def weibull_fading(beta=2.0, omega=1.0, n=1):
    """
    Convenience wrapper for Weibull fading.

    Returns
    -------
    np.ndarray
        Weibull fading envelope samples.
    """

    return FadingGenerator().weibull(beta=beta, omega=omega, n=n, return_complex=False)
