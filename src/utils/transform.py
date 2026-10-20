import os
import sys
from typing import List, Tuple, Union

import itk
import torch
import numpy as np
from tqdm import tqdm
import nibabel as nib
from nibabel.orientations import aff2axcodes

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utility import Timer, create_grid


class PointProcessor:
    def __init__(
        self,
        moving_path: str,
        fixed_path: str = None,
        transform_path: str = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        verbose: bool = False
    ) -> None:
        """Initialize the PointProcessor."""
        self.device = device
        self.verbose = verbose

        # Load moving image
        try:
            with Timer():
                print(f"Loading moving image from {moving_path}...")
                self.moving_image = itk.imread(moving_path)
                if self.verbose:
                    print(f"Moving image axcodes: {aff2axcodes(nib.load(moving_path).affine)}")
        except Exception as e:
            raise ValueError(f"Error reading the moving image from path '{moving_path}': {e}")

        # Get moving image meta data
        moving_meta = self._get_meta(self.device, self.verbose, fixed=False)
        self.moving_size, self.moving_spacing, self.moving_direction, self.moving_origin = moving_meta

        # Load fixed image
        if fixed_path is not None:
            try:
                with Timer():
                    print(f"Loading fixed image from {fixed_path}...")
                    self.fixed_image = itk.imread(fixed_path)
                    if self.verbose:
                        print(f"Fixed image axcodes: {aff2axcodes(nib.load(fixed_path).affine)}")
            except Exception as e:
                raise ValueError(f"Error reading the fixed image from path '{fixed_path}': {e}")

            # Get fixed image meta data
            fixed_meta = self._get_meta(self.device, self.verbose, fixed=True)
            self.fixed_size, self.fixed_spacing, self.fixed_direction, self.fixed_origin = fixed_meta

        # Load transform
        if transform_path is not None:
            try:
                with Timer():
                    print(f"Loading composite transform from {transform_path}...")
                    self.composite_transform = itk.transformread(transform_path)[0]
            except Exception as e:
                raise ValueError(f"Error reading the transform from path '{transform_path}': {e}")

    def _get_meta(
        self,
        device: str,
        verbose: bool = False,
        fixed: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the meta data of the image."""
        img = self.fixed_image if fixed else self.moving_image
        size = torch.tensor(img.GetLargestPossibleRegion().GetSize(), dtype=torch.int32, device=device)
        spacing = torch.tensor(img.GetSpacing(), dtype=torch.float32, device=device)
        origin = torch.tensor(img.GetOrigin(), dtype=torch.float32, device=device)
        direction = torch.tensor(np.asarray(img.GetDirection()).reshape(3, 3), dtype=torch.float32, device=device)

        if verbose:
            label = 'Fixed' if fixed else 'Moving'
            print(f"{label} image size: {size}")
            print(f"{label} image spacing: {spacing}")
            print(f"{label} image origin: {origin}")
            print(f"{label} image direction: {direction}", '\n')

        return size, spacing, direction, origin

    def _make_itk_continuous_index(
        self,
        continuous_index_point: List[float]
    ) -> itk.ContinuousIndex[itk.D, 3]:
        """Create an ITK continuous index object from a list of floats."""
        x = itk.ContinuousIndex[itk.D, 3]([0, 0, 0])  # Expects integer values when initialized with a sequence
        x[0], x[1], x[2] = continuous_index_point  # A quirk of ITK's Python wrapping implementation.
        return x

    def clamp_points(
        self,
        points: Union[List[itk.ContinuousIndex[itk.D, 3]], List[itk.Point[itk.D, 3]], torch.Tensor],
        clamp_limits: Tuple[List[int], List[int]]
    ) -> torch.Tensor:
        """Convert points to torch tensor and clamp them to the given limits."""
        if isinstance(points, torch.Tensor):
            return points.clamp(min=clamp_limits[0], max=clamp_limits[1])

        points = torch.tensor(points, dtype=torch.float32, device=self.device)
        return torch.stack([points[:, i].clamp(
            min=clamp_limits[0][i],
            max=clamp_limits[1][i])
            for i in range(3)], dim=1)

    def index2physical(
        self,
        index_points: torch.Tensor,
        from_fixed: bool,
        continuous: bool = False,
        use_itk: bool = False
    ) -> Union[List[itk.Point[itk.D, 3]], torch.Tensor]:
        """Convert index points to physical points in the given image space."""
        assert index_points.ndim == 2 and index_points.shape[1] == 3, "Expected a tensor of shape (N, 3)"

        if use_itk:
            print("Converting index points to physical points using ITK...")
            with Timer():
                img = self.fixed_image if from_fixed else self.moving_image

                if continuous:
                    physical_points = []
                    for index_point in tqdm(index_points):
                        continuous_index_point = self._make_itk_continuous_index(index_point.tolist())
                        physical_points.append(img.TransformContinuousIndexToPhysicalPoint(continuous_index_point))
                    return physical_points

                assert index_points.dtype == torch.int32, "Use index point of type int32 or set continuous=True"

            return [img.TransformIndexToPhysicalPoint(itk.Index[3](index_point.tolist())) for index_point in tqdm(index_points)]

        # Vectorised equivalent of img.Transform[Index|ContinuousIndex]ToPhysicalPoint
        print("Converting index points to physical points using tensor operations...")
        with Timer():
            if from_fixed:  # Avoids copying image data
                mat = self.fixed_direction @ torch.diag(self.fixed_spacing)  # 3 × 3
                return self.fixed_origin + (index_points.to(torch.float32) @ mat.T)

            mat = self.moving_direction @ torch.diag(self.moving_spacing)  # 3 × 3

        return self.moving_origin + (index_points.to(torch.float32) @ mat.T)

    # ------------------------------------------------------------------ #
    # Point transformation: three interchangeable backends.
    #
    #   "itk"  ITK's per-point loop; the reference implementation
    #          that the other backends are checked against.
    #   "cpu"  numpy, float64.
    #   "gpu"  torch float64 on CUDA.
    #   "auto" (default) "itk" below AUTO_MIN_POINTS points, where the loop
    #          is already fast; otherwise "gpu" if CUDA is available, else "cpu".
    #
    # "cpu" and "gpu" replicate ITK 5.3.0's CompositeTransform::TransformPoint
    # operation for operation, for composites of affine and displacement-field
    # stages, so their output is bit-identical to "itk":
    #   * the queue is applied back to front (CompositeTransform: rbegin..rend)
    #   * affine: Matrix * Point summed from 0.0 column by column, then + offset
    #   * field: continuous index = P2I * (p - origin); a point is inside if
    #     start - 0.5 <= index < end + 0.5 on every axis, and is returned
    #     unchanged otherwise (ImageFunction::IsInsideBuffer)
    #   * VectorLinearInterpolateImageFunction: ITK's Math::Floor (it differs
    #     from numpy's floor just below 0), the 8 neighbours in ITK's order
    #     with indices clamped to the buffer, overlaps multiplied x, y, z,
    #     zero overlaps skipped, and a stop once the summed overlap is 1.0
    #
    # On every call, a vectorised backend is first checked bit for bit against
    # "itk" on boundary probes of every stage and on a sample of the call's own
    # points; on any mismatch or error the call falls back to "itk".
    #
    # Select with the NODF_TRANSFORM_BACKEND environment variable or the
    # `backend` argument. It is not a command-line argument because
    # checkpoints store the full argument namespace, which `update_args`
    # restores, so a new argument would be missing from existing checkpoints.
    # ------------------------------------------------------------------ #

    class UnsupportedTransform(RuntimeError):
        """The composite holds a stage the vectorised backends do not replicate."""

    # Affine classes whose TransformPoint is MatrixOffsetTransformBase's
    # m_Matrix * p + m_Offset. Any other stage type is handled by "itk".
    _AFFINE_CLASSES = ("AffineTransform", "CenteredAffineTransform")

    # Below this many points "auto" uses "itk": the loop is already fast, and
    # comparable in cost to verifying a vectorised backend.
    AUTO_MIN_POINTS = 100_000

    # Number of each call's points checked bit for bit against "itk".
    VERIFY_SAMPLE = 20_000

    @staticmethod
    def _itk_floor(x: np.ndarray) -> np.ndarray:
        """ITK Math::Floor<int64>(double) on x86-64: RoundHalfIntegerToEven(2x - 0.5) >> 1.
        Not numpy's floor: it returns 0, not -1, for x in (-2**-55, 0)."""
        return np.rint(2.0 * x - 0.5).astype(np.int64) >> 1

    @staticmethod
    def _matvec_itk(M: np.ndarray, X: np.ndarray) -> np.ndarray:
        """ITK Matrix * Point/Vector: sum = 0.0, then sum += M[r, c] * x[c] in column order."""
        out = np.empty_like(X)
        for r in range(3):
            s = 0.0 + M[r, 0] * X[:, 0]
            s = s + M[r, 1] * X[:, 1]
            s = s + M[r, 2] * X[:, 2]
            out[:, r] = s
        return out

    @staticmethod
    def _matvec_itk_t(M: np.ndarray, X: torch.Tensor) -> torch.Tensor:
        """_matvec_itk on a float64 tensor. One op per step, so nothing is fused."""
        out = torch.empty_like(X)
        for r in range(3):
            s = 0.0 + float(M[r, 0]) * X[:, 0]
            s = s + float(M[r, 1]) * X[:, 1]
            s = s + float(M[r, 2]) * X[:, 2]
            out[:, r] = s
        return out

    @staticmethod
    def _bits_differ(a: np.ndarray, b: np.ndarray) -> int:
        """Number of differing 64-bit words. Stricter than ==: catches -0.0 vs 0.0."""
        a = np.ascontiguousarray(a, dtype=np.float64)
        b = np.ascontiguousarray(b, dtype=np.float64)
        return int((a.view(np.uint64) != b.view(np.uint64)).sum())

    @staticmethod
    def _itk_apply(tx, pts_np: np.ndarray) -> np.ndarray:
        """tx.TransformPoint on every row, exactly as the reference loop, without a progress bar."""
        out = np.empty_like(pts_np)
        for i, p in enumerate(pts_np):
            out[i] = tx.TransformPoint(p)
        return out

    def _decompose_composite(self) -> list:
        """The composite's stages as arrays, in application order."""
        c = self.composite_transform
        if c.GetNameOfClass() != "CompositeTransform":
            raise self.UnsupportedTransform(f"outer transform is {c.GetNameOfClass()}")
        stages = []
        for i in range(c.GetNumberOfTransforms()):
            sub = c.GetNthTransform(i)
            name = sub.GetNameOfClass()
            if name in self._AFFINE_CLASSES:
                a = itk.down_cast(sub)
                stages.append({
                    "kind": "affine", "itk": sub,
                    "M": np.asarray(itk.array_from_matrix(a.GetMatrix()), dtype=np.float64),
                    "offset": np.asarray(a.GetOffset(), dtype=np.float64),
                })
            elif name == "DisplacementFieldTransform":
                f = itk.down_cast(sub).GetDisplacementField()
                region = f.GetBufferedRegion()
                start = np.asarray(list(region.GetIndex()), dtype=np.int64)       # x, y, z
                end = start + np.asarray(list(region.GetSize()), dtype=np.int64) - 1
                I2P = (np.asarray(itk.array_from_matrix(f.GetDirection()), dtype=np.float64)
                       @ np.diag(np.asarray(list(f.GetSpacing()), dtype=np.float64)))
                stages.append({
                    "kind": "field", "itk": sub,
                    "field": np.asarray(itk.array_from_image(f), dtype=np.float64),  # z, y, x, 3
                    "origin": np.asarray(list(f.GetOrigin()), dtype=np.float64),
                    # ITK inverts with vnl's SVD, which its Python wrapping does not
                    # expose. np.linalg.inv agrees exactly for an identity geometry, as
                    # in uniGradICON's output; _verify_stages catches any disagreement.
                    "I2P": I2P, "P2I": np.linalg.inv(I2P),
                    "start": start, "end": end,
                    "start_ci": start - 0.5, "end_ci": end + 0.5,
                })
            else:
                raise self.UnsupportedTransform(
                    f"sub-transform {i} is {name}; the vectorised backends replicate only "
                    f"{', '.join(self._AFFINE_CLASSES)} and DisplacementFieldTransform")
        return stages[::-1]                       # CompositeTransform: rbegin -> rend

    def _transform_points_itk(self, pts_np: np.ndarray) -> np.ndarray:
        """Reference implementation: ITK, one point at a time. Do not optimise."""
        transformed = np.empty_like(pts_np)
        for i, p in enumerate(tqdm(pts_np)):
            point = self.composite_transform.TransformPoint(p)
            transformed[i] = point  # ITK Points can be assigned to numpy arrays directly
        return transformed

    def _field_stage_cpu(self, p: np.ndarray, st: dict) -> np.ndarray:
        """DisplacementFieldTransform::TransformPoint with its default linear interpolator."""
        out = p.copy()                                     # outside the buffer: unchanged
        idx = self._matvec_itk(st["P2I"], p - st["origin"])
        inside = np.all((idx >= st["start_ci"]) & (idx < st["end_ci"]), axis=1)
        if not inside.any():
            return out
        start, end, field = st["start"], st["end"], st["field"]
        ii = idx[inside]
        n = len(ii)
        base = self._itk_floor(ii)
        dist = ii - base.astype(np.float64)
        acc = np.zeros((n, 3))
        total = np.zeros(n)
        done = np.zeros(n, dtype=bool)
        for counter in range(8):                           # bit d set: upper neighbour on axis d
            overlap = np.ones(n)
            nb = [None, None, None]
            upper = counter
            for dim in range(3):
                if upper & 1:
                    nb[dim] = np.minimum(base[:, dim] + 1, end[dim])
                    overlap = overlap * dist[:, dim]
                else:
                    nb[dim] = np.maximum(base[:, dim], start[dim])
                    overlap = overlap * (1.0 - dist[:, dim])
                upper >>= 1
            active = (~done) & (overlap != 0)
            v = field[nb[2] - start[2], nb[1] - start[1], nb[0] - start[0], :]
            acc = np.where(active[:, None], acc + overlap[:, None] * v, acc)
            total = np.where(active, total + overlap, total)
            done = done | (total == 1.0)
        out[inside] = p[inside] + acc
        return out

    def _field_stage_gpu(self, p: torch.Tensor, g: dict) -> torch.Tensor:
        """_field_stage_cpu in torch; g comes from _stage_to_cuda."""
        out = p.clone()
        idx = self._matvec_itk_t(g["P2I"], p - g["origin"])
        inside = ((idx >= g["start_ci"]) & (idx < g["end_ci"])).all(dim=1)
        if not bool(inside.any()):
            return out
        start, end, field = g["start"], g["end"], g["field"]
        ii = idx[inside]
        n = ii.shape[0]
        f64 = dict(dtype=torch.float64, device=p.device)
        base = torch.round(2.0 * ii - 0.5).to(torch.int64) >> 1      # ITK Math::Floor
        dist = ii - base.to(torch.float64)
        acc = torch.zeros((n, 3), **f64)
        total = torch.zeros(n, **f64)
        done = torch.zeros(n, dtype=torch.bool, device=p.device)
        for counter in range(8):
            overlap = torch.ones(n, **f64)
            nb = [None, None, None]
            upper = counter
            for dim in range(3):
                if upper & 1:
                    nb[dim] = torch.clamp(base[:, dim] + 1, max=end[dim])
                    overlap = overlap * dist[:, dim]
                else:
                    nb[dim] = torch.clamp(base[:, dim], min=start[dim])
                    overlap = overlap * (1.0 - dist[:, dim])
                upper >>= 1
            active = (~done) & (overlap != 0)
            v = field[nb[2] - start[2], nb[1] - start[1], nb[0] - start[0], :]
            acc = torch.where(active[:, None], acc + overlap[:, None] * v, acc)
            total = torch.where(active, total + overlap, total)
            done = done | (total == 1.0)
        out[inside] = p[inside] + acc
        return out

    @staticmethod
    def _stage_to_cuda(st: dict) -> dict:
        """A stage's arrays as CUDA tensors."""
        f64 = dict(dtype=torch.float64, device="cuda")
        if st["kind"] == "affine":
            return {"kind": "affine", "M": st["M"], "offset": torch.as_tensor(st["offset"], **f64)}
        return {
            "kind": "field", "P2I": st["P2I"],
            "field": torch.as_tensor(st["field"], **f64),
            "origin": torch.as_tensor(st["origin"], **f64),
            "start_ci": torch.as_tensor(st["start_ci"], **f64),
            "end_ci": torch.as_tensor(st["end_ci"], **f64),
            "start": [int(v) for v in st["start"]],
            "end": [int(v) for v in st["end"]],
        }

    def _apply_stage_cpu(self, st: dict, P: np.ndarray) -> np.ndarray:
        if st["kind"] == "affine":
            return self._matvec_itk(st["M"], P) + st["offset"]
        return self._field_stage_cpu(P, st)

    def _apply_stage_gpu(self, g: dict, X: torch.Tensor) -> torch.Tensor:
        if g["kind"] == "affine":
            return self._matvec_itk_t(g["M"], X) + g["offset"]
        return self._field_stage_gpu(X, g)

    def _transform_points_cpu(self, pts_np: np.ndarray, stages: list = None,
                              chunk: int = 2_000_000) -> np.ndarray:
        """numpy float64, chunked to bound memory. Unverified: call through transform_physical_points."""
        stages = self._decompose_composite() if stages is None else stages
        out = np.empty_like(pts_np)
        for a in range(0, len(pts_np), chunk):
            p = pts_np[a:a + chunk]
            for st in stages:
                p = self._apply_stage_cpu(st, p)
            out[a:a + chunk] = p
        return out

    def _transform_points_gpu(self, pts_np: np.ndarray, stages: list = None,
                              chunk: int = 4_000_000) -> np.ndarray:
        """torch float64 on CUDA, chunked to bound memory. Unverified: call through
        transform_physical_points."""
        stages = self._decompose_composite() if stages is None else stages
        gpu_stages = [self._stage_to_cuda(st) for st in stages]
        out = np.empty_like(pts_np)
        for a in range(0, len(pts_np), chunk):
            p = torch.as_tensor(pts_np[a:a + chunk], dtype=torch.float64, device="cuda")
            for g in gpu_stages:
                p = self._apply_stage_gpu(g, p)
            out[a:a + chunk] = p.cpu().numpy()
        return out

    @staticmethod
    def _boundary_probes(st: dict) -> np.ndarray:
        """Points at and around every buffer boundary of a field stage, where the
        interpolation's edge cases live: the half-open domain, index clamping,
        and ITK's Floor just below 0."""
        axes = []
        for d in range(3):
            s, e = float(st["start"][d]), float(st["end"][d])
            vals = [s - 0.6, s - 0.5000001, s - 0.5, s - 0.4999999, s - 0.25,
                    s - 2.9e-17, s - 2.7e-17, s - 1e-17, s + 1e-17, s + 0.25, s + 0.5,
                    s + 1.0, 0.5 * (s + e) + 0.3, e - 0.5, e, e + 0.25, e + 0.4999999,
                    e + 0.5, e + 0.5000001]
            if s == 0.0:
                vals += [-0.0, 0.0]
            axes.append(vals)
        idx = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        # Random points on every boundary face as well: field values vary across
        # a face, and values near zero expose rounding-level differences.
        rng = np.random.default_rng(0)
        faces = []
        for d in range(3):
            s, e = float(st["start"][d]), float(st["end"][d])
            for v in (s - 0.5, s - 0.25, s - 2.7e-17, s + 0.25, e + 0.25, e + 0.4999999, e + 0.5):
                f = rng.uniform(st["start_ci"], st["end_ci"], (500, 3))
                f[:, d] = v
                faces.append(f)
        idx = np.vstack([idx] + faces)
        phys = idx @ st["I2P"].T + st["origin"]
        return np.vstack([phys, idx])         # raw indices as points too (keeps -0.0)

    def _verify_stages(self, backend: str, stages: list) -> Tuple[int, List[str]]:
        """Each stage alone must reproduce its ITK sub-transform bit for bit on probes."""
        rng = np.random.default_rng(0)        # local generator: global RNG state untouched
        n_probes, problems = 0, []
        for k, st in enumerate(stages):
            P = (self._boundary_probes(st) if st["kind"] == "field"
                 else rng.uniform(-200.0, 200.0, (2000, 3)))
            ref = self._itk_apply(st["itk"], P)
            if backend == "cpu":
                got = self._apply_stage_cpu(st, P)
            else:
                X = torch.as_tensor(P, dtype=torch.float64, device="cuda")
                got = self._apply_stage_gpu(self._stage_to_cuda(st), X).cpu().numpy()
            nd = self._bits_differ(ref, got)
            n_probes += len(P)
            if nd:
                problems.append(f"stage {k} ({st['kind']}): {nd} differing words on {len(P):,} probes")
        return n_probes, problems

    def _transform_points_verified(self, backend: str, pts_np: np.ndarray):
        """The vectorised backend's result, or None unless it reproduced itk bit for bit
        on the boundary probes of every stage and on a sample of these very points."""
        try:
            stages = self._decompose_composite()
            n_probes, problems = self._verify_stages(backend, stages)
            if problems:
                print(f"  [transform] {backend} backend failed verification: {problems[0]}; using itk")
                return None
            fn = self._transform_points_cpu if backend == "cpu" else self._transform_points_gpu
            with Timer():
                out = fn(pts_np, stages)
            sel = np.random.default_rng(0).choice(
                len(pts_np), size=min(len(pts_np), self.VERIFY_SAMPLE), replace=False)
            nd = self._bits_differ(self._itk_apply(self.composite_transform, pts_np[sel]), out[sel])
            if nd:
                print(f"  [transform] {backend} backend differed from itk in {nd} values of a "
                      f"{len(sel):,}-point sample; using itk")
                return None
            print(f"  [transform] {backend} backend: {len(pts_np):,} points, bit-identical to itk "
                  f"on {n_probes:,} boundary probes and a {len(sel):,}-point sample")
            return out
        except Exception as exc:              # any failure falls back to "itk"
            print(f"  [transform] {backend} backend raised {type(exc).__name__}: {exc}; using itk")
            return None
        finally:
            if backend == "gpu" and torch.cuda.is_available():
                torch.cuda.empty_cache()      # release cached GPU memory

    def _resolve_backend(self, backend: str, n_points: int) -> str:
        if backend == "auto":
            backend = (os.environ.get("NODF_TRANSFORM_BACKEND") or "auto").strip().lower()
        if backend == "auto":
            if n_points < self.AUTO_MIN_POINTS:
                return "itk"
            return "gpu" if torch.cuda.is_available() else "cpu"
        if backend not in ("itk", "cpu", "gpu"):
            raise ValueError(f"unknown transform backend {backend!r}; expected itk, cpu, gpu or auto")
        if backend == "gpu" and not torch.cuda.is_available():
            print("  [transform] CUDA unavailable; using the cpu backend")
            return "cpu"
        return backend

    def transform_physical_points(
        self,
        physical_points: Union[List[itk.Point[itk.D, 3]], torch.Tensor],
        backend: str = "auto",
    ) -> Union[List[itk.Point[itk.D, 3]], torch.Tensor]:
        """
        Transform the given physical points using the composite transform.

        backend: "itk" (the reference loop), "cpu", "gpu" or "auto"; see the
        comment above. A vectorised backend's result is used only after it
        reproduced "itk" bit for bit on this call; otherwise "itk" runs.
        """

        print("Transforming physical points...")

        if isinstance(physical_points, torch.Tensor):
            pts_np = physical_points.cpu().numpy().astype(np.float64)

            chosen = self._resolve_backend(backend, len(pts_np))
            transformed = None if chosen == "itk" else self._transform_points_verified(chosen, pts_np)
            if transformed is None:
                transformed = self._transform_points_itk(pts_np)

            # Pre-allocating a numpy array is faster than converting a list of ITK Point objects
            return torch.as_tensor(transformed, device=physical_points.device, dtype=physical_points.dtype)

        return [self.composite_transform.TransformPoint(physical_point) for physical_point in tqdm(physical_points)]

    def compare_backends(self, physical_points: torch.Tensor, n: int = 20000) -> dict:
        """QC: every available backend on the same points, against the itk reference.

        Returns {backend: {"differing_words": int, "max_abs": mm,
                           "probe_problems": [str], "seconds": s}}.
        A correct vectorised backend reports 0 differing words and no probe problems.
        """
        import time
        pts = physical_points.cpu().numpy().astype(np.float64)
        if len(pts) > n:
            pts = pts[np.sort(np.random.default_rng(0).choice(len(pts), n, replace=False))]
        t0 = time.time()
        ref = self._transform_points_itk(pts)
        result = {"itk": {"differing_words": 0, "max_abs": 0.0, "probe_problems": [],
                          "seconds": time.time() - t0}}
        stages = self._decompose_composite()
        for name in ["cpu"] + (["gpu"] if torch.cuda.is_available() else []):
            fn = self._transform_points_cpu if name == "cpu" else self._transform_points_gpu
            t0 = time.time()
            got = fn(pts, stages)
            result[name] = {
                "differing_words": self._bits_differ(ref, got),
                "max_abs": float(np.abs(ref - got).max()) if len(pts) else 0.0,
                "probe_problems": self._verify_stages(name, stages)[1],
                "seconds": time.time() - t0,
            }
        return result

    def get_min_max_moving_physical_vals(
        self,
        continuous: bool = False,
        boundary_only: bool = True,
        verbose: bool = False,
        use_itk: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the minimum and maximum physical values of the moving image."""

        print("==> Getting min and max physical values of the moving image...")

        index_points = create_grid(
            size=self.moving_size.tolist(),
            step=(1, 1, 1),
            device=self.device,
            verbose=True,
        )

        if boundary_only:
            # Keep only points on the surface: any point where x, y, or z is on the boundary
            mask = ((index_points == 0) | (index_points == self.moving_size - 1)).any(dim=1)
            index_points = index_points[mask]

        if verbose:
            # Print min and max values of the index points across all dimensions
            print("Index points min:", torch.min(index_points, dim=0).values)
            print("Index points max:", torch.max(index_points, dim=0).values, "\n")

        # Convert all indices to physical points
        physical_points = self.index2physical(
            index_points=index_points,
            from_fixed=False,
            continuous=continuous,
            use_itk=use_itk
        )

        if isinstance(physical_points, list):
            physical_points = torch.tensor(physical_points, dtype=torch.float32, device=self.device)

        min_physical_vals = torch.amin(physical_points, dim=0)
        max_physical_vals = torch.amax(physical_points, dim=0)
        assert min_physical_vals.shape == (3,) and max_physical_vals.shape == (3,), "Expected a tensor of shape (3,)"
        assert torch.all(max_physical_vals - min_physical_vals != 0), "Division by zero detected: Some dimensions have zero range"

        if verbose:
            print("Physical points min:", min_physical_vals)
            print("Physical points max:", max_physical_vals)
            print("Physical points range:", max_physical_vals - min_physical_vals, "\n")

        return min_physical_vals, max_physical_vals
