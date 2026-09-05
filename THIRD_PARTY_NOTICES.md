# Third-party notices

Metadata verification date: 2026-09-05. This notice describes candidate v8's
direct dependencies after the WO-01 corrections. It replaces the stale draft
dependency table; it is not a new software or scientific performance result.

## Direct dependencies

| Component / role | Version inspected | Licence evidence | Verification |
|---|---|---|---|
| NumPy / core | 2.2.6 | Installed distribution LICENSE.txt; project BSD-3-Clause terms | Verified locally |
| pandas / core | 2.3.3 | Installed distribution LICENSE; BSD 3-Clause | Verified locally |
| SciPy / core | 1.15.3 | Installed distribution LICENSE.txt; project BSD-3-Clause terms | Verified locally; bundled components have additional notices |
| tifffile / core | 2025.5.10 | Installed METADATA: License: BSD-3-Clause | Verified locally |
| h5py / hdf5 extra | 3.16.0 | Installed METADATA: License-Expression: BSD-3-Clause; full licence read | Verified locally |
| opencv-python / movie extra | 4.13.0.92 | METADATA says Apache 2.0; wrapper LICENSE.txt is MIT; LICENSE-3RD-PARTY.txt accompanies the wheel | Verified locally as a multi-component distribution, not a single blanket licence |
| WFDB / ecg extra | 4.1.2 | Versioned official PyPI metadata: MIT and MIT classifier | Verified remotely; no locally installed WFDB version or execution claimed |
| pytest / test extra | 9.1.1 | Installed METADATA: License-Expression: MIT; full licence read | Verified locally |
| Hypothesis / test extra | 6.138.15 | Installed METADATA: License-Expression: MPL-2.0 | Verified locally; not described as permissive |
| setuptools / build and test extra | 80.3.1 | Installed METADATA: License-Expression: MIT | Verified locally |
| wheel / build and test extra | Not installed in the inspected project environment | Local installed licence metadata unavailable | UNVERIFIED locally; verify the selected build environment before release |

Matplotlib and scikit-image are absent from candidate v8's direct dependency
declarations after WO-01 and are no longer listed as its direct runtime dependencies.
A dependency may itself depend on other packages; this direct-dependency inventory
is not a complete transitive software bill of materials.

## Resolution of the draft's unverified rows

The historical draft marked three rows unverified: scikit-image, h5py, and pytest.
After removing scikit-image from the release dependency set, the two applicable
unverified rows were h5py and pytest. Both were resolved by reading the installed
distribution METADATA and complete licence file:

- h5py 3.16.0: `h5py-3.16.0.dist-info/licenses/LICENSE`,
  SHA-256 `b773bc7cf038637a7fd8087648cc28e05f0f24d53770d1827147a40a1d268ab5`.
- pytest 9.1.1: `pytest-9.1.1.dist-info/licenses/LICENSE`,
  SHA-256 `ca836a5f9ecca3b2f350230faa20a48fb8b145653b5568d784862df864706b9b`.

WFDB evidence: https://pypi.org/pypi/wfdb/4.1.2/json.
No future dependency version is covered automatically by these observations.

The dependencies are installed separately; their source and binary notices must
be retained when those distributions are redistributed. NumPy, SciPy, h5py, and
OpenCV wheels may contain third-party native libraries with additional terms.
The MPL-2.0 Hypothesis test dependency does not justify a blanket statement that
every dependency is permissively licensed. This file records inspected evidence;
it does not settle institutional ownership or all downstream redistribution terms.

## Research tools and data outside this release

CaImAn/CNMF/OASIS, NAOMi, SimCalc/FISSA, and CASCADE scientific source are not
included in the bounded `audit_denoiser*` package. Their own upstream terms apply
to anyone obtaining and using those tools separately.

No raw movie, electrophysiology recording, model checkpoint, or research dataset
is licensed by this package's MIT file. DeepCAD-RT source-data provenance and
the simulator identities are documented in the research manuscript. CRCNS access
requires its own authorization and terms. PhysioNet records retain their source
licences and attribution requirements.

## Release boundary

Inspect the actual distribution and all native/transitive notices before publishing.
A local source tree, a wheel build, or this notice is not evidence of a public
licensed release or a completed third-party redistribution review.
