"""Population sampling and compiled C updates for the long-run GTFT experiment (Fig. S6).

Every step updates all agents at once from the same population state, clips strategies
to [epsilon, 1 - epsilon], and includes self-play. Gradients use a series approximation
(the moment expansion) when its error bound is small enough, and exact pairwise sums
otherwise; the bound covers each gradient, not the error built up over a whole run.
Each output file stores a hash of the C source, so a resumed run can check that it uses
the same code.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

C_SOURCE = r"""
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

/* Arrays are allocated once per chunk, never inside the timestep loop. */
int advance(double *a, const double *eta, const double *w, int n,
            int64_t steps, double b, double c, double eps, double tol) {
    double *buf = malloc((size_t)6*n*sizeof(double));
    if (!buf) return -1;
    double *r=buf, *u=r+n, *v=u+n, *z=v+n, *next=z+n;
    double mr[128], mq[128], mu[128];
    for (int64_t t=0; t<steps; ++t) {
        double lo=1., hi=-1., umax=0.;
        int isolated=0;
        for (int j=0; j<n; ++j) {
            r[j]=a[2*j]-a[2*j+1];
            u[j]=w[j]*(b*r[j]-c);
            if (r[j]<r[isolated]) isolated=j;
            umax=fmax(umax,fabs(b*r[j]-c));
        }
        /* Treat the lowest-r agent exactly. In the target population this
           separates the rare off-line agent from the narrowing main cluster. */
        for (int j=0; j<n; ++j) if (j!=isolated) {
            lo=fmin(lo,r[j]); hi=fmax(hi,r[j]);
        }
        double center=(hi+lo)*0.5, radius=(hi-lo)*0.5;
        double rho=0., dmin=1.;
        for (int i=0; i<n; ++i) {
            double d=1.-r[i]*center;
            v[i]=1./(d*d); z[i]=r[i]*radius/d;
            rho=fmax(rho,fabs(z[i])); dmin=fmin(dmin,d);
        }
        /* (1-ri*rj)^-2 = d^-2 sum_k (k+1) z_i^k x_j^k.
           |x_j|<=1. Each gradient numerator has absolute weighted sum
           <=2*umax. The omitted tail starts at k=terms. */
        int terms=1, cap=n/2 < 128 ? n/2 : 128;
        double power=rho;
        if (tol>0. && rho<1.) {
            while (terms<cap &&
                   2.*umax/(dmin*dmin)*power*((terms+1)-terms*rho)
                       /((1.-rho)*(1.-rho)) > tol) {
                ++terms; power*=rho;
            }
        } else terms=cap;
        if (terms<cap) {
            for (int k=0; k<terms; ++k) mr[k]=mq[k]=mu[k]=0.;
            for (int j=0; j<n; ++j) {
                if (j==isolated) continue;
                double x=radius>0. ? (r[j]-center)/radius : 0.;
                double uk=u[j];
                for (int k=0; k<terms; ++k) {
                    mr[k]+=uk*r[j]; mq[k]+=uk*a[2*j+1]; mu[k]+=uk;
                    uk*=x;
                }
            }
            for (int i=0; i<n; ++i) {
                double sr=0., sq=0., su=0.;
                for (int k=terms-1; k>=0; --k) {
                    sr=sr*z[i]+(k+1)*mr[k];
                    sq=sq*z[i]+(k+1)*mq[k];
                    su=su*z[i]+(k+1)*mu[k];
                }
                double p=a[2*i], q=a[2*i+1];
                double d=1.-r[i]*r[isolated], k=u[isolated]/(d*d);
                next[2*i]=p+eta[i]*(v[i]*(q*sr+sq)
                                      + k*(q*r[isolated]+a[2*isolated+1]));
                next[2*i+1]=q+eta[i]*(v[i]*(su-sq-p*sr)
                                      + k*(1.-a[2*isolated+1]-p*r[isolated]));
            }
        } else {
            for (int i=0; i<n; ++i) {
                double gp=0., gq=0., p=a[2*i], q=a[2*i+1];
                for (int j=0; j<n; ++j) {
                    double d=1.-r[i]*r[j], k=u[j]/(d*d);
                    gp+=k*(q*r[j]+a[2*j+1]);
                    gq+=k*(1.-a[2*j+1]-p*r[j]);
                }
                next[2*i]=p+eta[i]*gp; next[2*i+1]=q+eta[i]*gq;
            }
        }
        for (int i=0; i<2*n; ++i) {
            if (!isfinite(next[i])) { free(buf); return -2; }
            a[i]=fmin(1.-eps,fmax(eps,next[i]));
        }
    }
    free(buf); return 0;
}
"""


def compile_kernel(directory: Path, *, source: str = C_SOURCE):
    """Compile the C update code with $CC (default: cc) into a temporary library."""
    compiler = shutil.which(os.environ.get("CC", "cc"))
    if compiler is None:
        raise RuntimeError(
            "A C compiler is required: install one (macOS: xcode-select --install; "
            "Debian/Ubuntu: apt install build-essential) or set CC to its name."
        )
    source_path, library = directory / "update.c", directory / "update.so"
    source_path.write_text(source)
    subprocess.run(
        [compiler, "-O3", "-std=c99", "-shared", "-fPIC", str(source_path), "-o", str(library)]
        + ["-lm"],
        check=True,
        capture_output=True,
        text=True,
    )
    lib = ctypes.CDLL(str(library))
    array = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    lib.advance.argtypes = [array, array, array, ctypes.c_int, ctypes.c_int64] + [
        ctypes.c_double
    ] * 4
    lib.advance.restype = ctypes.c_int
    return lib


def advance(lib, agents, eta, weights, steps, b, c, epsilon, tolerance) -> None:
    """Advance agents in place; tolerance=0 selects direct pairwise summation."""
    status = lib.advance(agents, eta, weights, len(eta), steps, b, c, epsilon, tolerance)
    if status:
        raise RuntimeError(f"Compiled update failed (status {status}).")


def sample_population(agents, eta, size, epsilon, seed):
    """Keep every agent off the p = 1 - epsilon line and sample the agents on it.

    Agents on the line are sorted by q and split into consecutive groups of nearly equal
    size. One random agent from each group stands in for its group, with weight
    group size / N. Agents off the line come first and keep weight 1 / N each.
    """
    special = np.flatnonzero(agents[:, 0] < 1 - epsilon)
    ordinary = np.flatnonzero(agents[:, 0] >= 1 - epsilon)
    if not len(special) < size <= len(agents):
        raise ValueError(
            "Sample size must exceed the off-line count and not exceed population size."
        )
    ordinary = ordinary[np.argsort(agents[ordinary, 1], kind="stable")]
    strata = np.array_split(ordinary, size - len(special))
    rng = np.random.default_rng(seed)
    indices = np.array([*special, *(rng.choice(group) for group in strata)], dtype=np.int64)
    weights = np.array([*np.ones(len(special)), *(len(group) for group in strata)]) / len(agents)
    return agents[indices].copy(), eta[indices].copy(), weights, indices, len(special)


def validate_kernel(lib, agents, eta, weights, b, c, epsilon, tolerance):
    """Check 100 compiled steps (exact and approximate) against a plain NumPy calculation."""
    reference, exact, fast = (agents.copy() for _ in range(3))
    for _ in range(100):
        p, q = reference.T
        pi, qi, pj, qj = p[:, None], q[:, None], p[None, :], q[None, :]
        denom = (1 + pj * qi - qi * qj + pi * (-pj + qj)) ** 2
        common = c + b * (-pj + qj)
        gp = -((pj * qi + qj - qi * qj) * common) / denom
        gq = ((qj - 1) + pi * (pj - qj)) * common / denom
        reference += eta[:, None] * np.column_stack((gp @ weights, gq @ weights))
        np.clip(reference, epsilon, 1 - epsilon, out=reference)
    advance(lib, exact, eta, weights, 100, b, c, epsilon, 0.0)
    advance(lib, fast, eta, weights, 100, b, c, epsilon, tolerance)
    errors = np.array([np.max(np.abs(exact - reference)), np.max(np.abs(fast - reference))])
    if not np.all(np.isfinite(errors)) or np.max(errors) > 1e-10:
        raise RuntimeError(f"100-step dense reference validation failed: {errors}")
    return errors


def checkpoint_steps(steps: int) -> np.ndarray:
    """Log-spaced save points (16 per decade), plus one every million steps."""
    return np.unique(
        np.r_[
            np.geomspace(1, steps, 1 + int(np.ceil(16 * np.log10(steps)))).astype(np.int64),
            np.arange(1_000_000, steps, 1_000_000, dtype=np.int64),
            steps,
        ]
    )
